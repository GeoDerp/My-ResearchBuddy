import os
import random
import sys
import time
import urllib.robotparser
from typing import Annotated, List, TypedDict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(override=True)

# Flush all print() calls immediately even when stdout is piped.
import functools

print = functools.partial(print, flush=True)

# LangChain and LangGraph imports
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.types import Send

# ── Ethical Web Scraping ──────────────────────────────────────────────────────

# Honest identification — declare exactly what this bot is.
# Update the contact address before deploying in production.
BOT_USER_AGENT = (
    "DeepResearchBot/1.0 (Educational/Research Purpose; +contact: your-email@example.com)"
)

# Cache robots.txt parsers so each host is only queried once per run.
_robot_parsers: dict = {}


def can_fetch(url: str, user_agent: str) -> bool:
    """Returns True if robots.txt permits the given user agent to fetch the URL."""
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = f"{base_url}/robots.txt"

    if base_url not in _robot_parsers:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=5)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                _robot_parsers[base_url] = rp
            elif resp.status_code in (401, 403):
                # Server actively blocks access to robots.txt — assume blanket ban.
                _robot_parsers[base_url] = "BLOCKED"
            else:
                _robot_parsers[base_url] = rp
        except Exception:
            # robots.txt unavailable — web convention permits fetching.
            _robot_parsers[base_url] = rp

    parser = _robot_parsers[base_url]
    if parser == "BLOCKED":
        return False
    return parser.can_fetch(user_agent, url)


def fetch_webpage_text(url: str, max_chars: int = 8000) -> str:
    """Fetches a URL ethically: checks robots.txt, declares the bot honestly,
    applies a 1.5 s politeness delay, and strips boilerplate HTML tags."""
    if not can_fetch(url, BOT_USER_AGENT):
        return "[ETHICS BLOCK] robots.txt forbids automated access to this URL."

    try:
        # Randomized jitter prevents parallel agents from hitting the same server simultaneously.
        time.sleep(random.uniform(1.0, 3.0))
        response = requests.get(url, headers={"User-Agent": BOT_USER_AGENT}, timeout=10)

        if response.status_code in (401, 403):
            return f"[ETHICS BLOCK] Server refused connection (HTTP {response.status_code})."

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
            tag.extract()

        return soup.get_text(separator=" ", strip=True)[:max_chars]

    except requests.exceptions.RequestException as e:
        return f"Could not fetch content: {str(e)}"


# ── Reducer ───────────────────────────────────────────────────────────────────


def _add_or_reset(current: list, update) -> list:
    """Accumulates list updates from parallel nodes.
    Returns [] when update is None — used as a reset signal by node_refine_queries
    so the next fan-out iteration starts with a clean accumulator.
    operator.add cannot reset (add(existing, []) == existing), hence this custom reducer.
    """
    if update is None:
        return []
    return (current or []) + update


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class ResearchPlan(BaseModel):
    subtopics: List[str] = Field(
        description="Exactly 3 distinct, non-overlapping research subtopics derived from the query."
    )
    search_queries: List[str] = Field(
        description="One specific, targeted search query per subtopic, in the same order."
    )


class SubtopicRefinement(BaseModel):
    """Per-subtopic evaluation: a gap description and a thematically scoped refined query."""

    gap: str = Field(
        description=(
            "What is missing or inadequate in this specific subtopic's coverage. "
            "Empty string if the subtopic is sufficiently covered."
        )
    )
    query: str = Field(
        description=(
            "Refined search query STRICTLY within this subtopic's own thematic scope. "
            "Max 100 characters, plain natural language, no site: operators."
        )
    )


class EvaluationCriteria(BaseModel):
    is_complete: bool = Field(
        description="True only if the aggregated research fully resolves the original query."
    )
    feedback: str = Field(
        description="If is_complete is False, describe exactly what is missing or contradictory."
    )
    subtopic_refinements: List[SubtopicRefinement] = Field(
        description=(
            "Exactly 3 SubtopicRefinement objects, one per subtopic in the same order as the "
            "research plan. Each object's query is LOCKED to its subtopic's thematic scope — "
            "it must be clearly about that subtopic's theme, not about gaps in other subtopics. "
            "When is_complete is True, set gap='' and return the current query unchanged."
        )
    )


# ── State definitions ─────────────────────────────────────────────────────────


class ResearchState(TypedDict):
    original_query: str
    search_plan: str
    subtopics: List[str]
    search_queries: List[str]
    # Annotated fields — accumulated across parallel sub-agents; reset via None sentinel
    parallel_results: Annotated[List[str], _add_or_reset]
    sources: Annotated[List[str], _add_or_reset]
    evaluation_feedback: str
    loop_count: int
    is_complete: bool
    final_report: str


class SubtopicTask(TypedDict):
    """Payload carried by each Send object to the parallel sub-agent node."""

    subtopic: str
    query: str
    feedback: str


# ── Shared tools ──────────────────────────────────────────────────────────────

# DuckDuckGoSearchAPIWrapper returns structured results: [{"title", "link", "snippet"}]
# This gives us URLs for citation tracking, unlike DuckDuckGoSearchRun (text-only).
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=6)

# ── LLM auto-selection ────────────────────────────────────────────────────────
# Priority: OPENAI_API_KEY → MISTRAL_API_KEY. Set the key in your host shell and
# it is forwarded into the dev container automatically via remoteEnv in devcontainer.json.
# See README.md for setup instructions.
_openai_key = os.environ.get("OPENAI_API_KEY", "")
_mistral_key = os.environ.get("MISTRAL_API_KEY", "")

if _openai_key:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    _provider = "OpenAI · gpt-4o"
elif _mistral_key:
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(model="mistral-medium-3-5", temperature=0.2)
    _provider = "Mistral · mistral-medium-3-5"
else:
    llm = None
    _provider = None


# ── Nodes ─────────────────────────────────────────────────────────────────────


def node_analyze_problem(state: ResearchState) -> dict:
    """Supervisor node. Uses structured output to decompose the query into exactly
    3 subtopics with targeted search queries. These drive ALL downstream agents —
    nothing is hardcoded.
    """
    print("\n--- [NODE] ANALYZING PROBLEM & GENERATING RESEARCH PLAN ---")

    plan_llm = llm.with_structured_output(ResearchPlan)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert research architect operating within an autonomous, "
                "self-correcting investigation engine.\n\n"
                "STEP 1 — IDENTIFY THE QUERY DOMAIN from these categories:\n"
                "  software/engineering: framework comparisons, architecture decisions, DevOps, APIs, "
                "programming languages, system design, tooling\n"
                "  medical/biomedical: drugs, treatments, clinical research, health conditions, "
                "nutrition, longevity science, public health\n"
                "  consumer/hardware: device recommendations, audio or video equipment, component "
                "specs, product comparisons, microphones, headphones, cameras, peripherals\n"
                "  academic/science: physics, chemistry, biology, mathematics, pure research synthesis\n"
                "  business/market: strategy, market analysis, investment, industry trends, "
                "competitive landscape\n"
                "  general: history, policy, social science, law, or any multi-domain topic\n\n"
                "STEP 2 — DECOMPOSE into exactly 3 distinct subtopics using domain-appropriate angles.\n\n"
                "CRITICAL — Subtopic TITLES must be DOMAIN-SCOPED, not product-scoped:\n"
                "  CORRECT title: 'Authoritative architectural guidance for monolith decomposition'\n"
                "  WRONG title:   'Django.org official microservices migration documentation'\n"
                "Titles should describe the CATEGORY of knowledge to find, not the specific "
                "framework, product, or company. The search QUERY (STEP 3) carries the "
                "product/framework names for search precision. This keeps evaluation "
                "fair — a CNCF whitepaper satisfies 'authoritative architectural guidance' "
                "even if it does not mention Django by name.\n\n"
                "Domain-appropriate angles:\n\n"
                "  software/engineering angles:\n"
                "    [1] Authoritative foundations, standards, and architectural guidance\n"
                "        Target source types: cloud/infrastructure vendor docs (AWS, GCP, "
                "Azure, Kubernetes.io), CNCF/IETF standards publications, widely-cited "
                "architecture references (Martin Fowler, ThoughtWorks, High Scalability), "
                "GitHub READMEs of major open-source projects\n"
                "        QUERY STRATEGY for angle [1]: Do NOT include the framework name "
                "(Django, Rails, Flask, Laravel) in this query. Write queries focused on the "
                "PATTERN or CONCEPT name, and include known authoritative source names "
                "when appropriate. Examples that surface T1/T2 sources:\n"
                "  'Martin Fowler strangler fig monolith decomposition 2025'\n"
                "  'CNCF microservices migration patterns cloud native 2025'\n"
                "  'AWS Well-Architected microservices migration best practices'\n"
                "  'Kubernetes.io service mesh deployment patterns'\n"
                "These surface authoritative content; 'Django monolith migration 2025' "
                "surfaces tutorials only.\n"
                "    [2] Concrete named implementations, benchmarks, and comparative evaluations\n"
                "        Target source types: benchmark reports (TechEmpower, CNCF surveys), "
                "version-specific changelogs, performance blog posts, head-to-head comparisons\n"
                "    [3] Production practitioner experience and community consensus\n"
                "        Target source types: engineering blogs (Netflix, Shopify, Cloudflare, "
                "Martin Fowler, High Scalability blog), Stack Overflow, Hacker News discussions, "
                "CNCF annual survey reports, conference talk write-ups\n\n"
                "  medical/biomedical angles:\n"
                "    [1] Clinical evidence and mechanistic foundations\n"
                "        Target sources: PubMed search results, NIH pages, open-access journals, "
                "medRxiv/bioRxiv preprints, ClinicalTrials.gov summaries\n"
                "    [2] Approved interventions, guidelines, and current clinical practice\n"
                "        Target sources: FDA/EMA/TGA approval pages, WHO/CDC/NIH/NICE clinical "
                "guidelines, medical society position papers\n"
                "    [3] Real-world access, equity, implementation barriers, and cost considerations\n"
                "        Target sources: WHO essential medicines lists, generic drug pricing data, "
                "regulatory approval timelines across major health systems, patient "
                "advocacy organisations, cost-effectiveness analyses, health policy "
                "journals covering low-resource settings, medRxiv health policy preprints\n\n"
                "  consumer/hardware angles:\n"
                "    [1] Technical specifications and objective measured performance\n"
                "        Target sources: manufacturer spec sheets, AudioScienceReview, RTINGS.com, "
                "AnandTech, Tom's Hardware, PhoneArena, manufacturer datasheets\n"
                "        QUERY STRATEGY for angle [1]: Extract the KEY TECHNICAL REQUIREMENT "
                "from the user's query and include it as a search term. Examples: if the user "
                "asks for 'directional isolation', include 'cardioid' in the query; if they ask "
                "for 'USB', include 'USB microphone'. 'RTINGS cardioid gaming headset microphone "
                "under 150 2025' surfaces gaming headsets with directional mics. 'RTINGS best "
                "headset' misses the directionality requirement.\n"
                "    [2] Expert reviews and comparative rankings from specialist publications\n"
                "        Target sources: Wirecutter, RTINGS, What Hi-Fi, SoundOnSound, Head-Fi "
                "community wiki, Tom's Hardware, GSMArena, TechRadar\n"
                "    [3] Owner experience, long-term reliability, value analysis\n"
                "        Target sources: Reddit communities (r/headphones, r/audiophile, r/buildapc), "
                "retailer reviews, YouTube review channels, long-term ownership reports\n\n"
                "  academic/science angles:\n"
                "    [1] Established theory and foundational literature\n"
                "        Target sources: Wikipedia, arXiv, PubMed, Stanford/MIT OpenCourseWare, "
                "textbook summaries\n"
                "    [2] Current research frontiers and open questions\n"
                "        Target sources: recent arXiv/bioRxiv preprints 2024-2025, Nature/Science "
                "news, conference proceedings\n"
                "    [3] Practical applications, implications, and expert commentary\n"
                "        Target sources: science journalism, tech transfer reports, expert blogs\n\n"
                "  business/market angles:\n"
                "    [1] Market structure, quantitative data, and industry landscape\n"
                "        Target sources: Statista, industry association reports, SEC filings, "
                "financial journalism (FT, WSJ, Bloomberg)\n"
                "    [2] Strategic analysis and competitive dynamics\n"
                "        Target sources: HBR, McKinsey/BCG public reports, analyst notes\n"
                "    [3] Practitioner perspectives and real-world case studies\n"
                "        Target sources: founder interviews, investor analyses, company engineering "
                "blogs, case study repositories\n\n"
                "  general angles:\n"
                "    [1] Authoritative reference and established knowledge\n"
                "        Target sources: Wikipedia, government/NGO publications, encyclopaedias, "
                "official bodies\n"
                "    [2] Expert analysis and current evidence\n"
                "        Target sources: high-quality journalism, academic reviews, domain experts\n"
                "    [3] Community perspectives and real-world experience\n"
                "        Target sources: practitioner forums, user communities, recent commentary\n\n"
                "STEP 3 — WRITE SEARCH QUERIES following ALL of these rules:\n"
                "  - Under 120 characters total\n"
                "  - Plain natural language — NO site:, filetype:, inurl:, OR, AND, NOT operators\n"
                "  - Include 2025 or 2026 where recency matters for the domain\n"
                "  - Phrase each query to naturally surface the target sources for its angle\n"
                "  - Include technology/product/condition/concept names for specificity\n"
                "  - Queries describe the RESEARCH TOPIC, not documentation navigation. "
                "Do NOT construct queries that navigate to specific framework documentation "
                "(e.g., 'django.org microservices guide' or 'kubernetes.io service mesh tutorial'). "
                "Instead describe what you want to learn: 'Django microservices migration best "
                "practices 2025'. Exception: for consumer/hardware research, including specialist "
                "review database names as search terms IS correct and encouraged \u2014 "
                "e.g., 'RTINGS best headset microphone 2025' or 'Wirecutter podcasting "
                "headset under 150' are good queries.\n"
                "  - Do NOT embed exact constraint numbers from the original query (team sizes, "
                "exact request rates, precise dollar amounts, specific timelines) into the "
                "search query \u2014 these produce near-zero results. Use scale-class terms "
                "instead (e.g., 'large-scale', 'enterprise', 'high-traffic', 'budget "
                "microphone', 'cost effective') and let the query surface relevant evidence",
            ),
            ("human", "{query}"),
        ]
    )

    plan: ResearchPlan = (prompt | plan_llm).invoke({"query": state["original_query"]})

    plan_summary = "\n".join(
        f"  [{i + 1}] {t}: {q}" for i, (t, q) in enumerate(zip(plan.subtopics, plan.search_queries))
    )
    print(f"Research plan generated:\n{plan_summary}")

    return {
        "search_plan": plan_summary,
        "subtopics": plan.subtopics,
        "search_queries": plan.search_queries,
        "loop_count": state.get("loop_count", 0),
    }


def node_sub_agent(task: SubtopicTask) -> dict:
    """Parallel research worker. Each instance is dispatched independently via Send
    and handles exactly one subtopic. Receives only the SubtopicTask payload (not the
    full ResearchState). Return values are accumulated into ResearchState via
    the _add_or_reset reducer across all concurrent instances.
    """
    subtopic = task["subtopic"]
    query = task["query"]
    feedback = task.get("feedback", "")

    print(f"\n--- [SUB-AGENT] {subtopic} ---")
    print(f"  Query: {query[:90]}...")

    # Structured search — each result dict has "title", "link", "snippet".
    # max_results=4 balances depth against the politeness delay.
    # One retry with a 5 s backoff handles transient DuckDuckGo rate-limits.
    raw_results = []
    for _attempt in range(2):
        try:
            raw_results = search_wrapper.results(query, max_results=4)
            break
        except Exception as e:
            if _attempt == 0:
                print(f"  [DDG RETRY] Search failed ({e}), retrying in 5 s...")
                time.sleep(5)
            else:
                print(f"  [API ERROR] DuckDuckGo search failed after retry: {e}")

    if not raw_results:
        print(f"  [SKIP] No results for '{query[:80]}' — skipping LLM call.")
        return {
            "parallel_results": [f"### {subtopic}\n[NO RESULTS] No search results were returned for query: {query}"],
            "sources": [],
        }

    # Fetch full page content for each result while respecting robots.txt.
    citation_urls = []
    search_parts = []
    for r in raw_results:
        link = r.get("link", "")
        title = r.get("title", "Source")
        snippet = r.get("snippet", "")
        if link:
            citation_urls.append(f"[{title}]({link})")
            page_content = fetch_webpage_text(link)
        else:
            page_content = ""
        search_parts.append(
            f"Source: {link or 'N/A'}\nTitle: {title}\n"
            f"Search Snippet: {snippet}\nPage Content: {page_content}"
        )

    search_text = "\n\n---\n\n".join(search_parts)

    # Inject evaluator feedback into the LLM prompt — NOT the search query —
    # to avoid generating URLs that are too long for DuckDuckGo.
    feedback_instructions = ""
    if feedback and feedback not in ("None", ""):
        feedback_instructions = (
            f"\n\nPRIORITY FOCUS (ITERATION FEEDBACK):\n"
            f"The evaluation supervisor noted the following gaps in the overall research:\n"
            f'"{feedback}"\n'
            f"Pay special attention to extracting evidence that resolves this gap if present."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an autonomous research sub-agent gathering evidence for the "
                "subtopic: {subtopic}\n\n"
                "Analyse the search results and produce a structured, evidence-dense factual "
                "summary.\n\n"
                "SOURCE QUALITY — label each source you cite with its tier:\n"
                "  [T1 AUTHORITATIVE] Peer-reviewed publications, official docs (government, "
                "vendor, standards bodies), canonical measurement databases (RTINGS, "
                "AudioScienceReview, AnandTech, NIH/PubMed, kubernetes.io, django.org), "
                "ClinicalTrials.gov, FDA/EMA/WHO/CDC pages, CNCF publications\n"
                "  [T2 PRACTITIONER] Established engineering blogs (Netflix tech blog, Martin "
                "Fowler, Cloudflare blog, Shopify engineering), widely-read specialist "
                "publications (SoundOnSound, Ars Technica, IEEE Spectrum, Head-Fi, What Hi-Fi), "
                "Stack Overflow accepted answers, maintainer GitHub issues/discussions, CNCF "
                "survey reports, Wirecutter, RTINGS expert reviews\n"
                "  [T3 COMMUNITY] Individual blog posts, Reddit threads, Medium articles, forum "
                "posts — valuable for consensus signals; always note if single-source\n\n"
                "EXTRACTION PRIORITIES:\n"
                "  • Named entities: exact names, version numbers, model numbers\n"
                "  • Quantitative data: benchmarks, measurements, costs, percentages — any metric\n"
                "  • Procedural information: setup steps, configuration options, API patterns\n"
                "  • Comparative signals: A outperforms B on metric X — record both sides\n"
                "  • Limitations and trade-offs: failure modes, edge cases, licensing constraints\n"
                "  • Community consensus: multiple independent sources agreeing; flag single-source "
                "claims\n\n"
                "DATA AVAILABILITY — be honest about gaps:\n"
                "  If specific data is NOT in the search results (e.g., lab-measured polar plots, "
                "full-text RCT data behind paywalls, LMIC-specific deployment statistics, "
                "proprietary benchmarks, exact production-scale numbers), write:\n"
                "  [DATA GAP] <data type> — not publicly indexed; best available: <what was found>\n"
                "  This is a valid and important finding — do NOT fabricate specificity.\n\n"
                "Recency: prioritise 2024-2025 sources; explicitly note year for older data.\n"
                "Be analytical and neutral. Include counterpoints where they exist.\n"
                "Ethics: if a source returns '[ETHICS BLOCK]', rely ONLY on the Search Snippet."
                "{feedback_instructions}",
            ),
            (
                "human",
                "Search Results:\n\n{results}\n\nProvide your structured summary for: {subtopic}",
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "subtopic": subtopic,
            "results": search_text,
            "feedback_instructions": feedback_instructions,
        }
    )
    tagged_result = f"### {subtopic}\n{response.content}"

    return {
        "parallel_results": [tagged_result],
        "sources": citation_urls,
    }


def node_refine_queries(state: ResearchState) -> dict:
    """Executed on evaluation failure. Resets the parallel accumulators so the next
    fan-out starts with a clean slate. The refined search_queries were already written
    into state by node_evaluate_state, so no additional LLM call is needed here.
    """
    print("\n--- [NODE] REFINING — RESETTING ACCUMULATORS FOR NEXT ITERATION ---")
    print(f"  Feedback: {state.get('evaluation_feedback', '')}")
    print(f"  Refined queries: {state.get('search_queries', [])}")

    # Returning None triggers _add_or_reset → [] for both Annotated fields.
    return {
        "parallel_results": None,
        "sources": None,
    }


def node_evaluate_state(state: ResearchState) -> dict:
    """Self-correcting evaluation node. Cross-references all parallel research against
    the original constraints. On failure, writes LLM-generated refined queries into
    state["search_queries"] so the next fan-out uses better-targeted searches.
    """
    print("\n--- [NODE] EVALUATING HOLISTIC COMPLETENESS ---")

    eval_llm = llm.with_structured_output(EvaluationCriteria)
    aggregated = "\n\n".join(state.get("parallel_results", []))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a rigorous evaluation supervisor inside an autonomous research loop. "
                "Your role is a self-correcting stop-hook: determine whether the aggregated "
                "evidence is sufficient to answer the original query with high confidence, or "
                "whether the search loop must continue.\n\n"
                "HARD RULES \u2014 these override all other reasoning:\n"
                "  HARD RULE 1 \u2014 NO MEASUREMENTS: Objective acoustic or hardware "
                "measurements (polar plots, frequency response curves, SNR figures, THD+N, "
                "oscilloscope traces) are NEVER required. If any of your reasoning would fail "
                "a criterion because these lab measurements are absent, that reasoning is "
                "INVALID \u2014 discard it. Manufacturer-stated designations ('cardioid', "
                "'directional', 'noise-cancelling mic') are fully sufficient.\n"
                "  HARD RULE 2 \u2014 NO NICHE OVER-QUALIFICATION: Criterion 4 requires any "
                "community source on the TOPIC AREA; criterion 5 requires any source dated "
                "2024 or 2025. Do NOT add qualifiers like 'specifically about [exact niche]', "
                "'authoritative source for [niche]', or 'third-party validation of [spec]'. "
                "If your feedback contains those patterns, remove them before deciding.\n"
                "  HARD RULE 3 \u2014 MANUFACTURER CLAIMS SUFFICE: When manufacturer-stated "
                "polar pattern designations (cardioid, supercardioid, directional, "
                "hypercardioid) are present in the research for any identified product, "
                "criterion 1 for that technical constraint is SATISFIED. Do NOT require "
                "third-party lab validation of manufacturer claims.\n\n"
                "Mark is_complete = True if and only if ALL of the following hold:\n"
                "  1. Every explicit constraint in the original query (budget, technology stack, "
                "platform, use case, skill level, timeline, scale, accessibility, etc.) is "
                "addressed with specific, named answers — not vague generalities.\n"
                "     NOTE A \u2014 specific quantities: constraints expressed as exact numbers "
                "(requests per day, concurrent users, team size in engineers, budget amounts, "
                "timeline in months, etc.) are satisfied if the research addresses that scale "
                "class, use case, or range \u2014 exact numeric confirmation is NOT required. "
                "Evidence of 'millions of requests' satisfies '50M requests/day'; a case study "
                "for 'a large engineering team migrating a monolith' satisfies '20-engineer "
                "team'; pricing data showing 'under $200' satisfies 'under $150'.\n"
                "     NOTE B — LMIC/accessibility/equity: constraints about low-income countries, "
                "developing nations, underserved populations, or affordability are satisfied if "
                "(a) the research covers the global evidence base AND (b) includes at least one "
                "substantive analysis of cost, access, or equity barriers at any geographic "
                "scope. LMIC-specific pricing, country-level regulatory status, and local case "
                "studies are routinely absent from public web indexing — a [DATA GAP] notation "
                "acknowledging this absence fully satisfies the LMIC/accessibility constraint. "
                "IMPORTANT: when NOTE B applies, criteria 3, 4, and 5 are evaluated against the "
                "GLOBAL evidence base \u2014 a WHO global guideline satisfies criterion 3; any "
                "practitioner blog (regardless of country) satisfies criterion 4; any source "
                "from 2024-2025 satisfies criterion 5. Do NOT apply an 'LMIC-origin' filter "
                "to any criterion.\n"
                "     NOTE C — technical specs: constraints about technical characteristics "
                "(polar pattern, frequency response, SNR, latency, power draw) are satisfied if "
                "at least one source confirms the named product meets the requirement, even "
                "descriptively (e.g., 'cardioid pickup', 'low-latency mode') — objective "
                "measurement data is not required. IMPORTANT: if no product at the specified "
                "price point perfectly meets a technical spec (e.g., cardioid headset mics "
                "under $150 are dominated by gaming headsets with directional mics), the "
                "constraint is satisfied when the research: (a) identifies the closest "
                "available options by name and (b) explicitly explains the price/spec trade-off. "
                "Noting 'cardioid headset mics at this price are gaming headsets like HyperX X; "
                "purely omnidirectional headsets are unsuitable' satisfies criterion 1 for that "
                "technical constraint.\n"
                "     NOTE E \u2014 consumer spec terminology: For consumer/hardware queries, "
                "the following designations ALL satisfy a 'directional isolation' or 'cardioid' "
                "requirement: cardioid, supercardioid, directional, unidirectional, "
                "noise-cancelling microphone (when describing the microphone element). "
                "Gaming headsets marketed with 'directional' or 'noise-cancelling microphone' "
                "satisfy the directional isolation constraint even if the exact word 'cardioid' "
                "does not appear in search results. When the research identifies gaming headsets "
                "(HyperX, Razer, SteelSeries, Logitech G) with noise-cancelling microphones as "
                "the main sub-\u00a3150 options, this IS a valid answer to 'directional "
                "isolation' — do NOT fail criterion 1 solely because search snippets use "
                "'noise-cancelling' instead of 'cardioid'.\n"
                "  2. At least 2-3 concrete options, implementations, or approaches are "
                "identified and compared\n"
                "  3. At least one authoritative source appropriate to the query domain: "
                "for scientific or medical topics this means peer-reviewed papers or official "
                "guidelines; for software architecture and engineering topics, official vendor "
                "docs (IBM, AWS, GCP, Azure, Kubernetes.io), CNCF publications, widely-cited "
                "practitioner references (Martin Fowler, Netflix tech blog, Shopify engineering, "
                "ThoughtWorks, High Scalability, InfoQ, The New Stack, Increment) all qualify; "
                "for consumer products or market research, specialist publications (RTINGS, "
                "Wirecutter, AnandTech, manufacturer specs) fully qualify\n"
                "     NOTE D \u2014 software/engineering T3 fallback: DuckDuckGo frequently "
                "returns T3 practitioner content (Medium, personal blogs, tutorials) for "
                "well-established architectural patterns (strangler fig, saga, service mesh, "
                "Kubernetes, gRPC, CQRS) because these patterns originated before 2020 and "
                "their T1/T2 documentation does not always rank highly for recent-year queries. "
                "If the research contains substantively accurate and detailed coverage of the "
                "queried patterns/tools (correct descriptions of how they work, meaningful "
                "trade-offs, named real-world examples from recognisable companies), criterion 3 "
                "is satisfied for software/engineering even if the sources are primarily T3. "
                "Do NOT fail criterion 3 solely because martinfowler.com, CNCF.io, or "
                "Kubernetes.io were not in the returned search results.\n"
                "  4. At least one real-world or community source (blog post, forum thread, "
                "practitioner case study) is present\n"
                "  5. At least one source dated 2024 or 2025 is present\n"
                "  6. No CRITICAL contradiction is left unresolved. A contradiction is critical "
                "only if it leads to conflicting actionable recommendations (e.g., two sources "
                "say opposite things about which tool to use). Two surveys reporting different "
                "adoption percentages, or one source being more recent than another, is NOT a "
                "critical contradiction — note both data points and move on.\n\n"
                "Mark is_complete = False only when one of the above conditions is genuinely "
                "unmet. The following are NOT valid reasons to set is_complete=False:\n"
                "  - The research does not contain benchmarks AT the exact scale stated in the "
                "query (e.g., '50M requests/day'). If the evidence shows the technology handles "
                "that scale class or higher (e.g., 'millions of RPS', 'used at Netflix/Uber'), "
                "criterion 1 is satisfied — do NOT loop for scale-specific numbers.\n"
                "  - Examples come from a related language or framework (e.g., Java/Spring, "
                "Node.js, Go for a Django question). If the architectural or infrastructure "
                "pattern applies regardless of the specific language/framework "
                "(Kubernetes, service mesh, API gateway, saga pattern, event sourcing, "
                "distributed transactions, CQRS, strangler fig, vertical slice, circuit breaker, "
                "pub/sub messaging), the example fully satisfies criteria 1 AND 3 \u2014 do NOT "
                "require the named framework (Django, Rails, Laravel) to appear in the case "
                "study or documentation. NOTE: many web frameworks (Django, Flask, Rails, "
                "Laravel) do NOT publish official microservices or cloud architecture docs \u2014 "
                "this absence is expected and DOES NOT fail criterion 3. Cloud-native standards "
                "(CNCF, Kubernetes.io, AWS Well-Architected) and practitioner blogs (Martin "
                "Fowler, Netflix, Shopify, High Scalability) are the authoritative sources "
                "for architectural decisions regardless of the application framework.\n"
                "  - Two sources cite different statistics (e.g., different survey percentages). "
                "This is not an unresolved critical contradiction — cite both and move on.\n"
                "  - No peer-reviewed paper exists specifically about the query topic. For "
                "engineering/architecture topics, practitioner blogs, vendor docs, and company "
                "engineering case studies fully satisfy criterion 3.\n"
                "  - Hardware or acoustic measurement data (polar plots, frequency response "
                "curves, THD+N figures, oscilloscope traces) requires lab equipment and is "
                "rarely web-indexed, especially for budget products (typically under ~$200). "
                "Budget products are frequently not covered by measurement databases (RTINGS, "
                "AudioScienceReview). Criterion 3 is satisfied if (a) a manufacturer-stated "
                "designation is found (cardioid, directional, supercardioid, omnidirectional), "
                "OR (b) at least one expert or community review describes the product's "
                "relevant characteristic — polar diagrams or objective test data are NOT "
                "required and their absence is NOT a valid FAIL reason.\n"
                "  - LMIC-specific or region-specific quantitative data (country-level pricing, "
                "local regulatory approvals, LMIC clinical trial data, NGO deployment case "
                "studies) is routinely absent from public web indexing for novel interventions "
                "and products. Criterion 1 NOTE B already defines when the LMIC/accessibility "
                "constraint is satisfied. Criteria 3, 4, and 5 must NOT be applied with a "
                "'LMIC-origin' filter — a global WHO guidance, an international cost analysis, "
                "or a practitioner blog from any country satisfies those criteria. Do NOT loop "
                "demanding LMIC-specific authoritative sources, LMIC community blogs, or "
                "LMIC-dated 2025 sources separately.\n"
                "  - Full-text peer-reviewed papers are paywalled and not web-indexed. "
                "Open-access abstracts, preprints (arXiv, bioRxiv, medRxiv), NIH author "
                "manuscripts, and high-quality review summaries fully satisfy criterion 3 \u2014 "
                "do NOT loop demanding full-text access that a web search agent cannot obtain.\n"
                "  - When research contains '[DATA GAP]' markers, these represent valid research "
                "findings that establish specific data types are not publicly accessible via web "
                "search. [DATA GAP] markers are NOT evidence of insufficient research \u2014 they are "
                "honest coverage acknowledgments. A subtopic noting '[DATA GAP] LMIC-specific "
                "pricing not publicly indexed; best available: generic drug cost comparisons' is "
                "COMPLETE for that data type. Do NOT loop to find data established as unavailable "
                "by a [DATA GAP] marker.\n"
                "  - Criteria 4 and 5 have minimal threshold requirements — do NOT add "
                "specificity qualifiers beyond what the criterion text states. "
                "Criterion 4 requires 'any real-world or community source (blog post, "
                "forum thread, practitioner case study) on the topic area' — it does NOT "
                "require a source 'specifically about [exact product category + use case + "
                "price range]'. A Reddit thread about gaming headsets, a podcast equipment "
                "blog, or a home-studio forum discussion ALL satisfy criterion 4 for a "
                "headset-microphone query. Criterion 5 requires 'any source dated 2024 or "
                "2025' — it does NOT require that source to be authoritative (T1/T2) or to "
                "address the precise product combination in the query. A 2025 review article, "
                "forum post, or community discussion fully satisfies criterion 5. "
                "Do NOT write feedback such as 'no 2025-dated authoritative source "
                "specifically about X niche' or 'no community source specifically covering Y' "
                "\u2014 these are invalid fail reasons.\n\n"
                "SELF-CHECK before setting is_complete=False: Review each point in your "
                "proposed feedback. If it contains any of these prohibited patterns \u2014 "
                "'polar plot', 'frequency response', 'SNR', 'THD', 'objective measurement', "
                "'third-party validation', 'objective testing', 'specifically about', "
                "'no authoritative source for [niche]', 'no community source for [niche]' "
                "\u2014 that point is INVALID per the HARD RULES and NOT-valid-reasons above. "
                "Remove it. If removing all invalid points leaves no remaining gaps, set "
                "is_complete = True.\n\n"
                "When is_complete = False:\n"
                "- Identify only gaps that are realistically closable via general web search\n"
                "- APPLY NOTE A: do NOT write gaps about specific scale numbers "
                "(50M requests/day, 20-engineer team, etc.) being unconfirmed. Use "
                "scale-class terms in both gap descriptions and refinement queries.\n"
                "- Write a concise feedback string summarising the overall gaps\n"
                "- For subtopic_refinements, iterate the subtopic list in order and produce one "
                "SubtopicRefinement per subtopic:\n"
                "    gap: what is still missing in THIS subtopic's OWN coverage ('' if fine)\n"
                "    query: a search query STRICTLY within THIS subtopic's thematic scope\n"
                "  A gap that belongs to subtopic 3 affects ONLY subtopic 3's query — "
                "subtopics 1 and 2 must refine their own coverage from a different angle.",
            ),
            (
                "human",
                "Original Query:\n{query}\n\n"
                "Research Plan:\n{plan}\n\n"
                "Subtopics (produce subtopic_refinements in this exact order):\n{subtopics}\n\n"
                "Aggregated Research:\n{research}\n\n"
                "Does this fully meet all criteria? Provide subtopic_refinements if not.",
            ),
        ]
    )

    evaluation: EvaluationCriteria = (prompt | eval_llm).invoke(
        {
            "query": state["original_query"],
            "plan": state["search_plan"],
            "subtopics": "\n".join(
                f"  [{i}] {t}\n      current query: \"{q}\""
                for i, (t, q) in enumerate(
                    zip(state["subtopics"], state["search_queries"])
                )
            ),
            "research": aggregated,
        }
    )

    new_loop_count = state["loop_count"] + 1
    print(f"  Result: {'PASS' if evaluation.is_complete else 'FAIL — looping back'}")
    if not evaluation.is_complete:
        print(f"  Feedback: {evaluation.feedback}")

    # Failsafe — cap iterations to prevent runaway API costs.
    if new_loop_count >= 3:
        print("  WARNING: Max loops reached. Forcing completion with current best data.")
        evaluation.is_complete = True
        evaluation.feedback = "Max loops reached. Outputting best available synthesis."

    update: dict = {
        "is_complete": evaluation.is_complete,
        "evaluation_feedback": evaluation.feedback,
        "loop_count": new_loop_count,
    }

    # Write refined queries into state so node_refine_queries + next fan-out use them.
    if not evaluation.is_complete and evaluation.subtopic_refinements:
        n = min(len(evaluation.subtopic_refinements), len(state["subtopics"]))
        update["search_queries"] = (
            [r.query for r in evaluation.subtopic_refinements[:n]]
            + state["search_queries"][n:]
        )
        for i, r in enumerate(evaluation.subtopic_refinements[:n]):
            if r.gap:
                print(f"  [subtopic {i}] gap: {r.gap}")

    return update


def should_continue(state: ResearchState) -> str:
    return "synthesize" if state["is_complete"] else "refine"


def node_synthesize_report(state: ResearchState) -> dict:
    """Final synthesis node. Compiles all verified research into a comprehensive report.
    Every major claim is cited inline using markdown link format [Title](URL), and a
    References section is appended using the collected source URLs.
    """
    print("\n--- [NODE] SYNTHESIZING FINAL CITED REPORT ---")

    aggregated = "\n\n".join(state.get("parallel_results", []))
    sources_block = "\n".join(
        list(dict.fromkeys(state.get("sources", [])))
    )  # deduplicate, preserve order

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior research analyst producing the final deliverable of an "
                "autonomous, iterative research pipeline.\n\n"
                "Write a comprehensive, actionable report that fully resolves the user's query. "
                "Adapt the report structure to the question type while always including:\n\n"
                "Structure requirements:\n"
                "- Clear markdown headings and sub-headings appropriate to the content\n"
                "- A comparison or summary table when two or more options, tools, or approaches "
                "are evaluated\n"
                "- For implementation questions: a step-by-step guidance section covering setup, "
                "configuration, and integration\n"
                "- A trade-off section that candidly addresses limitations of the recommended "
                "approach\n"
                "- If the research contains [DATA GAP] markers, include a brief "
                "## Limitations & Data Availability section listing what data was not publicly "
                "accessible and what the best available proxy is — this adds credibility\n"
                "- A ## Final Recommendation section at the end — a definitive, opinionated "
                "answer that directly respects every constraint stated in the original query\n\n"
                "Citation requirements:\n"
                "- EVERY major claim, statistic, recommendation, or named entity MUST be cited "
                "inline using markdown: [Source Title](URL)\n"
                "- Use ONLY URLs from the provided source list — never fabricate or infer URLs\n"
                "- Conclude with a numbered References section listing every cited source\n\n"
                "Tone: analytical and precise. Avoid filler phrases. Treat the reader as an "
                "expert practitioner.",
            ),
            (
                "human",
                "Original Query:\n{query}\n\n"
                "Research Findings:\n{research}\n\n"
                "Available Sources for Citation:\n{sources}\n\n"
                "Write the final, fully cited report.",
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "query": state["original_query"],
            "research": aggregated,
            "sources": sources_block,
        }
    )

    return {"final_report": response.content}


# ── Edge functions ────────────────────────────────────────────────────────────


def dispatch_sub_agents(state: ResearchState) -> List[Send]:
    """Fans out one Send per subtopic, enabling parallel execution.
    Called from both 'plan' (initial dispatch) and 'refine' (loop-back dispatch).
    evaluation_feedback is embedded in each task so sub-agents can contextualise
    their refined queries without a separate re-planning LLM call.
    """
    feedback = state.get("evaluation_feedback", "")
    return [
        Send(
            "sub_agent",
            {
                "subtopic": subtopic,
                "query": query,
                "feedback": feedback,
            },
        )
        for subtopic, query in zip(state["subtopics"], state["search_queries"])
    ]


# ── Graph construction ────────────────────────────────────────────────────────


def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", node_analyze_problem)
    workflow.add_node("sub_agent", node_sub_agent)
    workflow.add_node("evaluate", node_evaluate_state)
    workflow.add_node("refine", node_refine_queries)
    workflow.add_node("synthesize", node_synthesize_report)

    # plan → fan-out to N parallel sub-agents (one per subtopic)
    workflow.set_entry_point("plan")
    workflow.add_conditional_edges("plan", dispatch_sub_agents)

    # parallel sub-agents → evaluate (LangGraph merges via _add_or_reset automatically)
    workflow.add_edge("sub_agent", "evaluate")

    # evaluate → done or loop
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {"synthesize": "synthesize", "refine": "refine"},
    )

    # refine → re-fan-out with refined queries and fresh accumulators
    workflow.add_conditional_edges("refine", dispatch_sub_agents)

    workflow.add_edge("synthesize", END)

    return workflow.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if llm is None:
        print("ERROR: No API key found.")
        print("Set OPENAI_API_KEY or MISTRAL_API_KEY in your host shell — see README.md.")
        exit(1)

    print(f"Provider : {_provider}")
    print("Initializing Deep Research Agent Graph...\n")
    agent = build_research_graph()

    if len(sys.argv) > 1:
        # Usage: python main.py "What is the best GPU for local LLMs in 2025?"
        # Unquoted multi-word args are also accepted: python main.py What is the best GPU...
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Research question: ").strip()

    if not query:
        print("ERROR: No query provided.")
        exit(1)

    initial_state: ResearchState = {
        "original_query": query,
        "search_plan": "",
        "subtopics": [],
        "search_queries": [],
        "parallel_results": [],
        "sources": [],
        "evaluation_feedback": "",
        "loop_count": 0,
        "is_complete": False,
        "final_report": "",
    }

    print("Kicking off research pipeline. This may take a few minutes as the agent iterates...")
    final_state = agent.invoke(initial_state)

    print("\n" + "=" * 50)
    print("                 FINAL REPORT")
    print("=" * 50 + "\n")
    print(final_state["final_report"])
    print("\n" + "=" * 50)

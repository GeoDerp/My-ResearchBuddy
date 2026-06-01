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


class EvaluationCriteria(BaseModel):
    is_complete: bool = Field(
        description="True only if the aggregated research fully resolves the original query."
    )
    feedback: str = Field(
        description="If is_complete is False, describe exactly what is missing or contradictory."
    )
    refined_queries: List[str] = Field(
        description=(
            "One refined search query per original subtopic, in the same order, each staying "
            "thematically aligned to its own subtopic while targeting the identified gap. "
            "Return the original queries unchanged if is_complete is True."
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
                "Decompose the user's query into exactly 3 distinct, non-overlapping subtopics "
                "that together fully resolve it. Choose angles appropriate to the query type "
                "(hardware recommendation, algorithm implementation, library comparison, research "
                "synthesis, security practice, etc.). As a universal guide, cover:\n"
                "  1. Theoretical / authoritative foundations — peer-reviewed research, official "
                "documentation, technical specifications, and current state-of-the-art from "
                "2024-2025 where possible\n"
                "  2. Concrete solutions and implementations — specific named tools, models, "
                "libraries, frameworks, products, or algorithms; include version numbers, "
                "benchmarks, and pricing where relevant\n"
                "  3. Real-world practitioner experience — how experts and communities actually "
                "use, deploy, and evaluate these solutions; developer blogs, forums, production "
                "case studies, and known pitfalls\n\n"
                "Rules for search queries:\n"
                "- Each query MUST be under 120 characters and phrased for a general web search engine\n"
                "- Include the current year (2025) to surface recent content\n"
                "- Do NOT use site: operators, boolean AND/OR/NOT, or quotation marks\n"
                "- Write natural-language queries a developer or researcher would type directly "
                "into a search engine",
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
                "Analyse the search results below and produce a structured, evidence-dense "
                "factual summary. The following rules apply universally regardless of domain:\n\n"
                "Extraction priorities:\n"
                "- Named entities: tool, model, library, framework, algorithm, or product names "
                "with exact version numbers\n"
                "- Quantitative data: performance benchmarks, latency, accuracy, cost, memory "
                "requirements, parameter counts — any measurable metric\n"
                "- Procedural information: setup steps, configuration options, API patterns, or "
                "integration requirements relevant to the subtopic\n"
                "- Comparative signals: where source A outperforms source B on a specific metric "
                "— record both sides\n"
                "- Limitations and trade-offs: known failure modes, edge cases, maintenance "
                "status, licensing constraints\n"
                "- Community consensus: note where multiple independent sources agree; flag "
                "single-source claims\n\n"
                "Recency: prioritise sources from 2024-2025; for older data, explicitly note "
                "the year so the synthesis stage can assess its currency.\n\n"
                "Be analytical and neutral — avoid promotional language and include "
                "counterpoints where they exist.\n\n"
                "Ethics: if a source returns '[ETHICS BLOCK]', do NOT guess its contents — "
                "rely ONLY on the Search Snippet provided for that source."
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
                "Mark is_complete = True if and only if ALL of the following hold:\n"
                "  1. Every explicit constraint in the original query (budget, technology stack, "
                "platform, use case, skill level, timeline, etc.) is addressed with specific, "
                "named answers — not vague generalities\n"
                "  2. At least 2-3 concrete options, implementations, or approaches are "
                "identified and compared\n"
                "  3. At least one authoritative source appropriate to the query domain: "
                "for scientific or medical topics this means peer-reviewed papers or official "
                "guidelines; for consumer products, technology, or market research, specialist "
                "publications (e.g., RTINGS, Wirecutter, SoundGuys, AnandTech, manufacturer "
                "specs) fully qualify as authoritative\n"
                "  4. At least one real-world or community source (blog post, forum thread, "
                "practitioner case study) is present\n"
                "  5. At least one source dated 2024 or 2025 is present\n"
                "  6. No critical contradiction in the evidence is left unresolved\n\n"
                "Mark is_complete = False only when one of the above conditions is genuinely "
                "unmet — do not fail for aspirational depth, perfect completeness, or the "
                "absence of peer-reviewed papers on topics where such literature does not "
                "commonly exist (e.g., consumer hardware reviews, practitioner workflows).\n\n"
                "When is_complete = False:\n"
                "- Identify only gaps that are realistically closable via general web search\n"
                "- Write a concise feedback string naming the specific, searchable gap\n"
                "- Provide exactly 3 refined search queries following the slot mapping shown. "
                "Each refined_queries[N] is LOCKED to its subtopic — the subtopic's scope "
                "takes priority over gap coverage. If a gap belongs to subtopic 3, only "
                "subtopic 3's query targets it; subtopics 1 and 2 refine their OWN existing "
                "coverage instead. Under 100 characters, plain natural language, no site: "
                "operators or boolean syntax.",
            ),
            (
                "human",
                "Original Query:\n{query}\n\n"
                "Research Plan:\n{plan}\n\n"
                "Required refined_queries slots — each slot is locked to its subtopic:\n{subtopics}\n\n"
                "Aggregated Research:\n{research}\n\n"
                "Does this fully meet all criteria? Provide refined queries if not.",
            ),
        ]
    )

    evaluation: EvaluationCriteria = (prompt | eval_llm).invoke(
        {
            "query": state["original_query"],
            "plan": state["search_plan"],
            "subtopics": "\n".join(
                f'  refined_queries[{i}] → "{t}" | current query: "{q}"'
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
    if not evaluation.is_complete and evaluation.refined_queries:
        n = min(len(evaluation.refined_queries), len(state["subtopics"]))
        update["search_queries"] = evaluation.refined_queries[:n] + state["search_queries"][n:]

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

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


def _merge_dicts(current: dict, update: dict) -> dict:
    """Merges dictionary updates from parallel sub-agents."""
    if not isinstance(current, dict):
        current = {}
    if not isinstance(update, dict):
        return current
    return {**current, **update}


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class ResearchPlan(BaseModel):
    subtopics: List[str] = Field(
        description="Exactly 3 distinct, non-overlapping research subtopics derived from the query.",
        min_length=3,
        max_length=3,
    )
    search_queries: List[str] = Field(
        description="One specific, targeted search query per subtopic, in the same order.",
        min_length=3,
        max_length=3,
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
    quality_score: float = Field(
        description="Holistic quality of the aggregated evidence, 1–10. 7.5+ means sufficiently comprehensive for synthesis."
    )
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
    subtopic_gaps: List[str]
    # Annotated fields — accumulated across parallel sub-agents
    parallel_results: Annotated[dict, _merge_dicts]
    sources: Annotated[dict, _merge_dicts]
    evaluation_feedback: str
    loop_count: int
    prev_quality_score: float
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
                "You are an expert research architect in an autonomous investigation engine.\n\n"
                "<RULES>\n"
                "1. Decompose the query into exactly 3 MECE (Mutually Exclusive, Collectively Exhaustive) subtopics.\n"
                "2. Subtopic TITLES must outline the *category of knowledge* to discover (e.g., 'Authoritative architectural patterns', 'Clinical evidence guidelines'), not just restate product names. Keep exact product/framework targets in the search query.\n"
                "3. Ensure the 3 subtopics span these distinct domain-agnostic research angles:\n"
                "   [Angle A] Theoretical/Authoritative: Official docs, foundational standards, peer-reviewed literature, official guidelines or textbooks.\n"
                "   [Angle B] Empirical/Objective: Benchmarks, measurable performance, clinical trials, quantitative market data, or head-to-head comparisons.\n"
                "   [Angle C] Practitioner/Community: Real-world case studies, expert reviews, integration challenges, community forums, or field experience.\n"
                "4. Search queries must be < 120 chars, plain natural language. NO advanced search operators (site:, OR, AND) as they degrade API performance.\n"
                "5. Include the current year where recency matters. Map exact constraints to qualitative scale-class terms (e.g., use 'enterprise high-traffic' instead of '50M requests/day').\n"
                "</RULES>\n",
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
        "subtopic_gaps": [""] * len(plan.subtopics),
        "loop_count": state.get("loop_count", 0),
    }


def node_sub_agent(task: SubtopicTask) -> dict:
    """Parallel node: researches a single subtopic and returns structured evidence."""
    subtopic = task["subtopic"]
    query = task["query"]
    feedback = task.get("feedback", "")

    print(f"  [Sub-Agent] Searching: '{query}'")
    raw_results = []
    for _attempt in range(2):
        try:
            raw_results = search_wrapper.results(query, max_results=4)
            break
        except Exception as e:
            import time

            if _attempt == 0:
                print(f"  [DDG RETRY] Search failed ({e}), retrying in 5 s...")
                time.sleep(5)

    if not raw_results:
        print(f"  [SKIP] No results for '{query[:80]}' — skipping LLM call.")
        return {
            "parallel_results": {
                subtopic: f"### {subtopic}\n[NO RESULTS] No search results were returned for query: {query}"
            },
            "sources": {subtopic: []},
        }

    formatted_results = "\n\n---\n\n".join(
        [
            f"Source [{i + 1}]: {res.get('url', 'Unknown URL')}\n"
            f"Title: {res.get('title', 'No Title')}\n"
            f"Content:\n{res.get('content', '')}"
            for i, res in enumerate(raw_results)
        ]
    )

    feedback_instructions = ""
    if feedback:
        feedback_instructions = (
            f"\nPREVIOUS LOOP FEEDBACK TO ADDRESS:\n"
            f"The evaluator noted this gap: '{feedback}'\n"
            "Ensure you explicitly look for and extract evidence addressing this gap.\n"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an autonomous research sub-agent. Gather evidence for the subtopic: {subtopic}\n\n"
                "Label every cited source with its tier:\n"
                "  [T1 AUTHORITATIVE] Official docs, peer-reviewed publications, standards bodies\n"
                "  [T2 PRACTITIONER] Established blogs, specialist reviews, engineering write-ups\n"
                "  [T3 COMMUNITY] Reddit, forums, individual posts\n\n"
                "Extract:\n"
                "  • Named entities, version numbers, model numbers, specific dates\n"
                "  • Quantitative data: benchmarks, measurements, costs, percentages\n"
                "  • Comparative signals, trade-offs, and known limitations\n\n"
                "If data is absent: [DATA GAP] <type> — not indexed; best proxy: <what was found>\n"
                "Be analytical and neutral. Note counterpoints where evidence conflicts.\n"
                "{feedback_instructions}",
            ),
            (
                "human",
                "Search Results:\n\n{results}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "subtopic": subtopic,
            "feedback_instructions": feedback_instructions,
            "results": formatted_results,
        }
    )

    citation_urls = [res.get("url") for res in raw_results if res.get("url")]
    tagged_result = f"### {subtopic}\n{response.content}"

    return {
        "parallel_results": {subtopic: tagged_result},
        "sources": {subtopic: citation_urls},
    }


def node_refine_queries(state: ResearchState) -> dict:
    """Pass-through node. State merging now ensures completed subtopics are retained."""
    print("\n--- [NODE] REFINING — PREPARING NEXT ITERATION ---")
    print(f"  Feedback: {state.get('evaluation_feedback', '')}")
    print(f"  Refined queries: {state.get('search_queries', [])}")
    return {}


def node_evaluate_state(state: ResearchState) -> dict:
    """Reads the accumulated parallel_results and decides if research is complete."""
    print("\n--- [NODE] EVALUATING HOLISTIC COMPLETENESS ---")

    eval_llm = llm.with_structured_output(EvaluationCriteria)
    parallel_results = state.get("parallel_results", {})
    aggregated = "\n\n".join(parallel_results.get(t, "") for t in state.get("subtopics", []))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a rigorous evaluation supervisor inside an autonomous research loop. "
                "Your role is a self-correcting stop-hook: determine whether the aggregated "
                "evidence is sufficient to answer the original query with high confidence.\n\n"
                "Mark is_complete = True if and only if ALL of the following hold:\n"
                "  1. Every explicit constraint in the original query is addressed with specific, named answers.\n"
                "     NOTE: Exact numeric confirmation is NOT required if the evidence addresses that scale/budget class.\n"
                "  2. At least 2-3 concrete options, implementations, or approaches are compared.\n"
                "  3. At least one authoritative source appropriate to the inferred domain.\n"
                "  4. At least one real-world or community source is present.\n"
                "  5. At least one recent source (e.g., from the last 1-2 years) is present.\n"
                "  6. No critical contradiction is left unresolved without acknowledging both sides.\n\n"
                "Assign quality_score (1–10) for the holistic evidence quality:\n"
                "  1–4: Thin, mostly irrelevant, or severely incomplete\n"
                "  5–6: Partial — some constraints addressed, key data missing\n"
                "  7–8: Comprehensive — major constraints met, minor gaps acceptable for synthesis\n"
                "  9–10: Exceptional — quantitative data, authoritative sources, no meaningful gaps\n\n"
                "IMPORTANT: [DATA GAP] markers represent valid research findings that data is not publicly accessible. "
                "Do NOT loop to find data established as unavailable by a [DATA GAP] marker.\n\n"
                "When is_complete = False:\n"
                "- Identify only gaps that are realistically closable via general web search.\n"
                "- Write a concise feedback string summarising the overall gaps.\n"
                "- Provide a subtopic_refinements array with EXACTLY one SubtopicRefinement per subtopic, matching the same order.\n"
                "- For subtopics that are complete, leave the gap empty ('') and reuse the current query.\n"
                "- For subtopics that need more research, write a specific gap and a targeted new query.",
            ),
            (
                "human",
                "Original Query:\n{query}\n\n"
                "Aggregated Evidence from sub-agents:\n{evidence}\n\n"
                "Review the evidence against the query. Is it fully answered?",
            ),
        ]
    )
    chain = prompt | eval_llm
    evaluation: EvaluationCriteria = chain.invoke(
        {"query": state["original_query"], "evidence": aggregated}
    )

    score = evaluation.quality_score
    loop = state.get("loop_count", 0)
    prev_score = state.get("prev_quality_score", 0.0)

    # Smart stop conditions — evaluated before logging
    is_complete = evaluation.is_complete
    if not is_complete:
        if score >= 7.5:
            is_complete = True
            print(f"  Score {score:.1f} \u2265 7.5 \u2014 accepting as sufficient")
        elif loop >= 1 and (score - prev_score) < 1.0:
            is_complete = True
            print(f"  Score delta {score - prev_score:.1f} < 1.0 \u2014 diminishing returns, forcing complete")
        elif loop >= 3:
            is_complete = True
            print("  WARNING: Max loops reached. Forcing completion with current best data.")

    print(f"  Result: {'PASS' if is_complete else 'FAIL \u2014 looping back'}")
    print(f"  Score: {score:.1f}/10")

    gaps_found = [r.gap for r in evaluation.subtopic_refinements if r.gap]
    calculated_feedback = "; ".join(gaps_found) if gaps_found else "No gaps."

    if not is_complete:
        print(f"  Feedback: {calculated_feedback}")

    update = {
        "is_complete": is_complete,
        "evaluation_feedback": calculated_feedback,
        "loop_count": loop + 1,
        "prev_quality_score": score,
    }

    if not is_complete:
        n = len(state["subtopics"])
        update["search_queries"] = [r.query for r in evaluation.subtopic_refinements[:n]] + state[
            "search_queries"
        ][n:]
        update["subtopic_gaps"] = [r.gap for r in evaluation.subtopic_refinements[:n]] + [""] * (
            len(state["subtopics"]) - n
        )
        for i, r in enumerate(evaluation.subtopic_refinements[:n]):
            if r.gap:
                print(f"  [subtopic {i}] gap: {r.gap}")
                print(f"               query: {r.query}")

    return update


def should_continue(state: ResearchState) -> str:
    return "synthesize" if state["is_complete"] else "refine"


def node_synthesize_report(state: ResearchState) -> dict:
    """Final synthesis node. Compiles all verified research into a comprehensive report.
    Every major claim is cited inline using markdown link format [Title](URL), and a
    References section is appended using the collected source URLs.
    """
    print("\n--- [NODE] SYNTHESIZING FINAL CITED REPORT ---")

    aggregated = "\n\n".join(state.get("parallel_results", {}).values())
    sources_dict = state.get("sources", {})
    all_urls = []
    for urls in sources_dict.values():
        all_urls.extend(urls)
    sources_block = "\n".join(list(dict.fromkeys(all_urls)))

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
    Only dispatches for subtopics that have a reported gap, saving API calls.
    """
    global_feedback = state.get("evaluation_feedback", "")
    gaps = state.get("subtopic_gaps", [""] * len(state["subtopics"]))

    tasks = []
    for subtopic, query, gap in zip(state["subtopics"], state["search_queries"], gaps):
        if state.get("loop_count", 0) == 0 or gap:
            tasks.append(
                Send(
                    "sub_agent",
                    {
                        "subtopic": subtopic,
                        "query": query,
                        "feedback": gap if gap else global_feedback,
                    },
                )
            )
    return tasks


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
    import sys

    default_query = (
        "Compare microservices vs monolith architecture for a new enterprise banking application"
    )
    query = sys.argv[1] if len(sys.argv) > 1 else default_query

    initial_state: ResearchState = {
        "original_query": query,
        "search_plan": "",
        "subtopics": [],
        "search_queries": [],
        "subtopic_gaps": [],
        "parallel_results": {},
        "sources": {},
        "evaluation_feedback": "",
        "loop_count": 0,
        "prev_quality_score": 0.0,
        "is_complete": False,
        "final_report": "",
    }

    graph = build_research_graph()
    config = {"recursion_limit": 15}

    print(f"\n=== STARTING AUTONOMOUS RESEARCH ===\nQuery: {query}\n")
    try:
        final_state = graph.invoke(initial_state, config=config)
        print("\n\n========================================================")
        print("                 FINAL RESEARCH REPORT                  ")
        print("========================================================\n")
        print(final_state.get("final_report", "No report generated."))
    except Exception as e:
        print(f"\n\n[FATAL ERROR] Pipeline terminated:\n{e}")

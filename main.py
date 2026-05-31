import os
import operator
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field

# LangChain and LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END
from langgraph.types import Send


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
            "One refined search query per original subtopic to address the identified gaps. "
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

    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.2)
    _provider = "Mistral · mistral-large-latest"
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

    # On loop-back iterations, append a brief excerpt of the evaluator's feedback.
    # Feedback is truncated to avoid generating search URLs that exceed length limits.
    effective_query = query
    if feedback and feedback not in ("None", ""):
        brief_feedback = feedback[:200].rsplit(" ", 1)[0]  # trim at word boundary
        effective_query = f"{query} {brief_feedback}"

    print(f"\n--- [SUB-AGENT] {subtopic} ---")
    print(f"  Query: {effective_query[:90]}...")

    # Structured search — each result dict has "title", "link", "snippet"
    raw_results = search_wrapper.results(effective_query, max_results=6)

    # Build citation strings from result URLs for inclusion in the final report.
    citation_urls = [
        f"[{r.get('title', 'Source')}]({r.get('link', '')})" for r in raw_results if r.get("link")
    ]

    search_text = "\n\n".join(
        f"Source: {r.get('link', 'N/A')}\nTitle: {r.get('title', '')}\n{r.get('snippet', '')}"
        for r in raw_results
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
                "counterpoints where they exist.",
            ),
            (
                "human",
                "Search Results:\n\n{results}\n\nProvide your structured summary for: {subtopic}",
            ),
        ]
    )

    response = (prompt | llm).invoke({"subtopic": subtopic, "results": search_text})
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
                "  3. At least one authoritative source (peer-reviewed paper, official "
                "documentation, or recognised technical expert) is present\n"
                "  4. At least one real-world or community source (blog post, forum thread, "
                "practitioner case study) is present\n"
                "  5. At least one source dated 2024 or 2025 is present\n"
                "  6. No critical contradiction in the evidence is left unresolved\n\n"
                "Mark is_complete = False only when one of the above conditions is genuinely "
                "unmet — do not fail for aspirational depth or perfect completeness.\n\n"
                "When is_complete = False:\n"
                "- Write a concise feedback string that names the specific gap (e.g. "
                "'Missing: concrete pricing for option B')\n"
                "- Provide exactly one refined search query per subtopic — STRICTLY under 100 "
                "characters, plain natural language, no site: operators or boolean syntax — "
                "targeting only the identified gap",
            ),
            (
                "human",
                "Original Query:\n{query}\n\n"
                "Research Plan:\n{plan}\n\n"
                "Aggregated Research:\n{research}\n\n"
                "Does this fully meet all criteria? Provide refined queries if not.",
            ),
        ]
    )

    evaluation: EvaluationCriteria = (prompt | eval_llm).invoke(
        {
            "query": state["original_query"],
            "plan": state["search_plan"],
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
    sources_block = "\n".join(state.get("sources", []))

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

    query = """
    I am a solo software developer with a budget of AUD $3,000. I want to run large language
    models locally for daily coding assistance, RAG experimentation, and occasional fine-tuning.
    Based on the latest 2024-2025 research, Reddit community recommendations (r/LocalLLaMA,
    r/MachineLearning), and technical blogs:
      1. What hardware should I buy — GPU, CPU, RAM, and storage — and why?
      2. What software stack should I use — inference framework, quantisation format, and
         fine-tuning tooling?
      3. Which specific open-source LLM models are best suited to this setup?
    Provide specific product names, current AUD prices, VRAM requirements, and benchmark
    comparisons. Prioritise value-for-money and developer ergonomics over raw performance.
    """

    initial_state: ResearchState = {
        "original_query": query.strip(),
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

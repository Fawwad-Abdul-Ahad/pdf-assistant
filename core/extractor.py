from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def extract_insights(summary: str) -> dict:
    """
    Extracts actionable items, decisions, and questions
    from the video summary using parallel LLM chains.

    Args:
        summary: The final summary text from summarize_transcript()

    Returns:
        A dict with keys: action_items, decisions, questions
    """

    llm = ChatMistralAI(model="labs-leanstral-1-5-1")
    parser = StrOutputParser()

    # ─── Prompt 1: Actionable Items ────────────────────────────────────
    action_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert analyst.
Extract all ACTIONABLE ITEMS from the given text.
Actionable items are tasks or steps that someone needs to DO.

Return as a numbered list. If none found, say 'No actionable items found.'
Do not add anything that is not in the text."""
        ),
        ("human", "{summary}")
    ])

    # ─── Prompt 2: Decisions ────────────────────────────────────────────
    decision_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert analyst.
Extract all DECISIONS made or discussed in the given text.
Decisions are conclusions, choices, or agreements reached.

Return as a numbered list. If none found, say 'No decisions found.'
Do not add anything that is not in the text."""
        ),
        ("human", "{summary}")
    ])

    # ─── Prompt 3: Questions ────────────────────────────────────────────
    question_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert analyst.
Extract all QUESTIONS raised or left unanswered in the given text.
These are open questions, doubts, or things needing further discussion.

Return as a numbered list. If none found, say 'No questions found.'
Do not add anything that is not in the text."""
        ),
        ("human", "{summary}")
    ])

    # ─── Build Individual Chains ────────────────────────────────────────
    action_chain   = action_prompt   | llm | parser
    decision_chain = decision_prompt | llm | parser
    question_chain = question_prompt | llm | parser

    # ─── RunnableParallel — All 3 run at the same time ─────────────────
    parallel_chain = RunnableParallel(
        action_items = action_chain,
        decisions    = decision_chain,
        questions    = question_chain
    )

    result = parallel_chain.invoke({"summary": summary})

    return result
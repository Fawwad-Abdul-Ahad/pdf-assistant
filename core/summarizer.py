from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv

load_dotenv()


def summarize_transcript(transcript: str) -> str:
    """
    Splits transcript into chunks, summarizes each chunk (Map),
    then combines and generates a final summary (Reduce).

    This prevents token overflow for long video transcripts.

    Args:
        transcript: Full transcribed text from the video.

    Returns:
        A final structured summary string.
    """

    llm = ChatMistralAI(model="labs-leanstral-1-5-1")
    parser = StrOutputParser()

    # ─── Step 1: Split transcript into manageable chunks ───────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(transcript)
    print(f"Transcript split into {len(chunks)} chunks.")

    # ─── Step 2: MAP — Summarize each chunk individually ───────────────
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant.
Summarize the following portion of a video transcript.
Be concise. Keep only the important points.
Do not add anything that is not in the text."""
        ),
        (
            "human",
            "Transcript portion:\n\n{chunk}"
        )
    ])

    map_chain = map_prompt | llm | parser

    mini_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        mini_summary = map_chain.invoke({"chunk": chunk})
        mini_summaries.append(mini_summary)

    combined = "\n\n".join(mini_summaries)

    # ─── Step 3: REDUCE — Final summary from all mini summaries ────────
    reduce_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert video summarizer.
You are given multiple partial summaries of a video transcript.
Combine them into one final, structured summary.

Format:
- **Main Topic**: What is the video about?
- **Key Points**: Most important points (bullet list)
- **Conclusion**: Final takeaway"""
        ),
        (
            "human",
            "Partial summaries:\n\n{combined}"
        )
    ])

    reduce_chain = reduce_prompt | llm | parser

    print("Generating final summary...")
    final_summary = reduce_chain.invoke({"combined": combined})

    return final_summary

def generate_title(final_summary: str) -> str:
    """
    Generate a title from the transcription.

    Args:
        transcription: Full transcribed text from the video.

    Returns:
        A title string.
    """
    llm = ChatMistralAI(model="labs-leanstral-1-5-1")
    parser = StrOutputParser()

    chain = RunnablePassthrough() | RunnableLambda(lambda x: {"transcription": x}) | ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert video summarizer.
You are given a transcription of a video.
Generate a title for the video.

Format:
- **Title**: A short, catchy title for the video"""
        ),
        (
            "human",
            "Transcription:\n\n{transcription}"
        )
    ]) | llm | parser

    title = chain.invoke({"transcription": final_summary})

    return title
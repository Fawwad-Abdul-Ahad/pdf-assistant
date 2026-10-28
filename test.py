

from utils.audio_processor import process_audio, cleanup
from core.summarizer import summarize_transcript, generate_title
from core.transcribe import transcribe_all_chunks
from core.extractor import extract_insights
from dotenv import load_dotenv
load_dotenv()

def test():
    """test the whole process"""
    audio_chunks = process_audio("https://www.youtube.com/watch?v=MP415FWqCgE")
    transcription = transcribe_all_chunks(audio_chunks, translation=True)
    summary = summarize_transcript(transcription)
    title = generate_title(summary)
    insights = extract_insights(summary)
    cleanup(audio_chunks)
    print(summary)
    print(insights)
    print(title)
    
test()
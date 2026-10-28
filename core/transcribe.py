from utils.load_model import load_model


def transcribe_chunk(
    chunk_path: str,
    translation: bool = False
) -> str:
    """
    Transcribe one audio chunk using Groq.

    translation=False:
        Original spoken language is preserved.

    translation=True:
        Audio is translated into English.
    """

    print(f"Transcribing: {chunk_path}")

    client = load_model()

    with open(chunk_path, "rb") as file:

        if translation:

            result = client.audio.translations.create(
                file=file,
                model="whisper-large-v3",
                response_format="text",
                temperature=0
            )

        else:

            result = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                response_format="text",
                temperature=0
            )

    return result.strip()


def transcribe_all_chunks(
    audio_chunks: list[str],
    translation: bool = False
) -> str:
    """
    Transcribe all audio chunks and combine
    them into one transcript.
    """

    transcription_parts = []

    for index, chunk in enumerate(audio_chunks, start=1):

        print(f"\nChunk {index}/{len(audio_chunks)}")

        text = transcribe_chunk(
            chunk_path=chunk,
            translation=translation
        )

        if text:
            transcription_parts.append(text)

    return "\n".join(transcription_parts)



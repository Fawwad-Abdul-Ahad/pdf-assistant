import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_audio_youtube(url : str) -> str:
    """download the audio or tracribe the audio from the url"""
    output_path = os.path.join(DOWNLOAD_DIR,("%(title)s.%(ext)s"))
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'restrictfilenames': True,
        'noplaylist': True,
        "postprocessors" :[
            {
                "key" :"FFmpegExtractAudio",
                "preferredcodec" : "wav",
                "preferredquality" : "192",
            }    
        ],
        "quiet" : True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm",".wav").replace(".mp4",".wav")
    return filename


def convert_to_wav(input_path : str) -> str:
    """convert the audio in to standard 16 khz whisper"""
    ouput_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(ouput_path, format="wav")
    return ouput_path


def chunk_vdeo (wav_path : str, chunk_mint : int = 10 )-> list:
    """Chunking the audio in to smaller chunks"""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_mint * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start+chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_audio(source : str) -> list:
    """download the audio, convert it to standard 16 khz whisper, chunk it in to smaller chunks"""  
    if source.startswith("http://") or source.startswith("https://"):
        filename = download_audio_youtube(source)
    else:
        filename = source
    wav_path = convert_to_wav(filename)
    chunks = chunk_vdeo(wav_path)       
    return chunks

import os
import shutil

def cleanup(file_paths: list):
    """
    Delete all temporary audio files after processing.
    """
    for filename in os.listdir(DOWNLOAD_DIR):
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    print("Downloads folder cleaned.")

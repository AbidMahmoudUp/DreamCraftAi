import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from pathlib import Path
import tempfile
import base64
from utils.logging import setup_logging

logger = setup_logging()

class AudioService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()

    def speech_to_text(self, audio_path: str) -> str:
        with sr.AudioFile(audio_path) as source:
            audio_data = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio_data, language="en-US")
            except sr.UnknownValueError:
                raise ValueError("Could not understand audio")
            except sr.RequestError as e:
                raise RuntimeError(f"Speech recognition error: {str(e)}")

    def text_to_speech(self, text: str, output_path: str) -> str:
        self.tts_engine.save_to_file(text, output_path)
        self.tts_engine.runAndWait()
        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def validate_audio(self, audio_file: bytes, content_type: str) -> str:
        if content_type != "audio/wav":
            raise ValueError("Only WAV audio supported")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_file)
            tmp_path = tmp_audio.name
        audio = AudioSegment.from_wav(tmp_path)
        if len(audio) < 500:
            raise ValueError("Audio too short (min 0.5 seconds)")
        return tmp_path
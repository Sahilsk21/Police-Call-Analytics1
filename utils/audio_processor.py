import os
import subprocess
import warnings
import torch
from pydub import AudioSegment
from transformers import pipeline
from typing import Dict, Optional
import json
try:
    import whisper
except ImportError:
    raise ImportError("Whisper not installed. Please run: pip install openai-whisper")

# Global models to avoid reloading on every AudioProcessor call
WHISPER_MODEL = None
TRANSLATOR_PIPELINE = None

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        model_size = "base" if not torch.cuda.is_available() else "small"
        WHISPER_MODEL = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    return WHISPER_MODEL

def get_translation_pipeline():
    global TRANSLATOR_PIPELINE
    if TRANSLATOR_PIPELINE is None:
        TRANSLATOR_PIPELINE = pipeline(
            task="translation",
            model="facebook/nllb-200-distilled-600M",
            device=0 if torch.cuda.is_available() else -1
        )
    return TRANSLATOR_PIPELINE

class AudioProcessor:
    def __init__(self):
        self._verify_system_dependencies()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcriber = get_whisper_model()
        self.translator = get_translation_pipeline()

    def _verify_system_dependencies(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "FFmpeg not found. Install it with:\n"
                "Linux: sudo apt install ffmpeg\n"
                "Mac: brew install ffmpeg\n"
                "Windows: Download from ffmpeg.org"
            ) from e

    def _convert_to_wav(self, audio_path: str) -> str:
        if audio_path.endswith(".wav"):
            return audio_path
        try:
            audio = AudioSegment.from_file(audio_path)
            wav_path = os.path.splitext(audio_path)[0] + ".converted.wav"
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed: {str(e)}") from e

    def transcribe_and_translate(self, audio_path: str) -> Dict[str, Optional[str]]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav_path = self._convert_to_wav(audio_path)

        try:
            result = self.transcriber.transcribe(wav_path)
            transcript = result["text"]
            lang = result["language"]

            if lang != 'en':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    translated = self.translator(transcript)
                return {
                    "original_text": transcript,
                    "original_lang": lang,
                    "translated_text": translated[0]['translation_text'],
                    "translation": True
                }
            else:
                return {
                    "translated_text": transcript,
                    "original_lang": lang,
                    "translation": False
                }
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e
        finally:
            if wav_path.endswith("converted.wav"):
                try:
                    os.remove(wav_path)
                except:
                    pass

# For testing standalone
if __name__ == "__main__":
    try:
        processor = AudioProcessor()
        result = processor.transcribe_and_translate("police_call.mp3")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


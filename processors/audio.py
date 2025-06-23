import requests
import tempfile
from config import HF_CONFIG

class AudioProcessor:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def _call_hf_api(self, endpoint, data=None, files=None):
        response = requests.post(
            endpoint,
            headers=self.headers,
            data=data,
            files=files
        )
        return response.json()

    def transcribe(self, audio_bytes):
        """Use HF Whisper API"""
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp.seek(0)
            return self._call_hf_api(
                HF_CONFIG["whisper"]["api"],
                files={"file": tmp}
            )

    def translate(self, text, source_lang):
        """Use HF Translation API"""
        return self._call_hf_api(
            HF_CONFIG["translation"]["api"],
            json={"inputs": text}
        )

    def process(self, audio_bytes):
        """Complete audio processing pipeline"""
        # Step 1: Transcription
        transcript_result = self.transcribe(audio_bytes)
        transcript = transcript_result.get("text", "")
        
        if not transcript:
            return {"text": "", "translated": False}
        
        # Step 2: Language detection (simple heuristic)
        is_english = all(ord(c) < 128 for c in transcript)
        
        # Step 3: Translation if needed
        if not is_english:
            translation = self.translate(transcript, "auto")
            return {
                "text": translation[0]["translation_text"],
                "original": transcript,
                "translated": True
            }
        
        return {
            "text": transcript,
            "translated": False
        }

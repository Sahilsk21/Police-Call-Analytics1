import os

# Hugging Face configuration
HF_CONFIG = {
    "whisper": {
        "api": "https://api-inference.huggingface.co/models/openai/whisper-base",
        "local_fallback": False
    },
    "translation": {
        "api": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-mul-en",
        "local_fallback": False
    },
    "ner": {
        "api": "https://api-inference.huggingface.co/models/dslim/bert-base-NER",
        "local_fallback": False
    }
}

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB limit
TEMP_DIR = "./temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

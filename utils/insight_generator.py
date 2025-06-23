from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
from functools import lru_cache

CONFIG_FILE = Path(__file__).parent / "../crime_categories.json"

class CrimeClassifier:
    def __init__(self):
        self.model = self._load_model()
        self.categories = self._load_categories()
        self.category_embeddings = self._load_cached_embeddings()

    def _load_model(self):
        print("Loading embedding model...")
        return SentenceTransformer('all-MiniLM-L6-v2')

    def _load_categories(self):
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "Robbery": "Illegal taking of property through force or threat",
            "Assault": "Physical attack or violent contact",
            "Cybercrime": "Computer/internet-based illegal activities",
            "Burglary": "Unauthorized entry to commit theft",
            "Vandalism": "Deliberate property destruction",
            "Harassment": "Unwanted persistent behavior causing distress",
            "Fraud": "Deception for personal gain",
            "Kidnapping": "Unlawful taking and confinement of a person",
            "Arson": "Deliberate setting of fires to property",
            "DrugOffense": "Illegal drug-related activities",
            "Other": "General complaint not matching specific categories"
        }

    def _save_categories(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, indent=2, ensure_ascii=False)

    def _load_cached_embeddings(self):
        descriptions = list(self.categories.values())
        print("Encoding category descriptions...")
        return self.model.encode(descriptions, show_progress_bar=False)

    @lru_cache(maxsize=1000)
    def classify(self, text, threshold=0.4):
        text_embedding = self.model.encode([text], show_progress_bar=False)
        similarities = cosine_similarity(text_embedding, self.category_embeddings)
        scores = similarities[0]
        max_idx = np.argmax(scores)

        best_category = list(self.categories.keys())[max_idx]
        confidence = scores[max_idx]

        return (best_category, confidence) if confidence >= threshold else ("Other", confidence)

    def update_category(self, name, description):
        self.categories[name] = description
        self._save_categories()
        self.category_embeddings = self._load_cached_embeddings()
        self.classify.cache_clear()

    def remove_category(self, name):
        if name in self.categories and name != "Other":
            del self.categories[name]
            self._save_categories()
            self.category_embeddings = self._load_cached_embeddings()
            self.classify.cache_clear()
            return True
        return False

# Singleton-like global instance
_classifier_instance = None

def get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = CrimeClassifier()
    return _classifier_instance

def classify_crime(text, threshold=0.4):
    return get_classifier().classify(text, threshold)

def update_crime_categories(updates):
    classifier = get_classifier()
    for name, desc in updates.items():
        if desc is not None:
            classifier.update_category(name, desc)
        else:
            classifier.remove_category(name)

def get_current_categories():
    return get_classifier().categories.copy()


import re
import requests
from transformers import pipeline
from config import HF_CONFIG
from utils.cache import load_model

class NLPProcessor:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {self.hf_token}" if hf_token else {}}
        
        # Load models
        self.weapon_classifier = load_model(
            "weapon_classifier",
            lambda: pipeline("zero-shot-classification", 
                           model="facebook/bart-large-mnli")
        )
        
        # Enhanced weapon categories
        self.weapon_categories = [
            "gun", "knife", "firearm", "handgun", "rifle",
            "shotgun", "blunt object", "sharp object", "explosive"
        ]

    def extract_entities(self, text):
        """Enhanced entity extraction for police calls"""
        results = {
            "locations": set(),
            "times": set(),
            "suspects": set(),
            "weapons": set()
        }

        # 1. Extract locations with multiple methods
        self._extract_locations(text, results)
        
        # 2. Extract time references
        self._extract_times(text, results)
        
        # 3. Extract suspects with context awareness
        self._extract_suspects(text, results)
        
        # 4. Detect weapons with zero-shot and regex
        self._detect_weapons(text, results)

        return {k: list(v) for k, v in results.items()}

    def _extract_locations(self, text, results):
        """Multi-method location extraction"""
        # Method 1: Address patterns "1330 Alpha Sayo Day"
        address_pattern = r'\d{3,4}\s+[A-Z][A-Za-z\s\-]+(?:\s+(?:street|st|avenue|ave|road|rd))?'
        addresses = re.findall(address_pattern, text)
        results["locations"].update(addresses)
        
        # Method 2: Business names  "Pete's coffee"
        business_pattern = r'(?:at|in)\s+([A-Z][A-Za-z0-9\'\s]+(?:store|shop|coffee|bar|restaurant|market))'
        businesses = re.findall(business_pattern, text, re.IGNORECASE)
        results["locations"].update(businesses)
        
        # Method 3: NER API fallback
        try:
            response = requests.post(
                HF_CONFIG["ner"]["api"],
                headers=self.headers,
                json={"inputs": text}
            )
            for entity in response.json():
                if entity["entity_group"] in ["LOC", "GPE", "FAC"]:
                    results["locations"].add(entity["word"])
        except Exception as e:
            print(f"NER API error: {e}")

    def _extract_times(self, text, results):
        """Extract absolute and relative time references"""
        # Relative times  "right now"
        relative_times = re.findall(
            r'(?:right\s+now|currently|at\s+this\s+time|just\s+now)',
            text, re.IGNORECASE
        )
        results["times"].update(rt.lower() for rt in relative_times)
        
        # Clock times  "3:45 PM"
        clock_times = re.findall(
            r'\d{1,2}:\d{2}\s*(?:AM|PM)?',
            text, re.IGNORECASE
        )
        results["times"].update(clock_times)

    def _extract_suspects(self, text, results):
        """Context-aware suspect extraction"""
        # Pattern 1: Suspect descriptions  "white male about 40"
        desc_pattern = r'(white|black|hispanic|asian)\s+(male|female)\s+(?:about|approximately)?\s*(\d{2})?'
        for race, gender, age in re.findall(desc_pattern, text, re.IGNORECASE):
            desc = f"{race.lower()} {gender.lower()}"
            if age:
                desc += f" ~{age}"
            results["suspects"].add(desc)
        
        # Pattern 2: Names in suspect context
        suspect_pattern = r'(suspect|shooter|attacker|perpetrator|intruder)\s+(?:is\s+)?([A-Z][a-z]+)'
        for _, name in re.finditer(suspect_pattern, text, re.IGNORECASE):
            results["suspects"].add(name)
        
        # Pattern 3: Clothing descriptions  "wearing red hat"
        clothing_pattern = r'(?:wearing|has)\s+(?:a\s+)?([a-z]+\s+(?:hat|jacket|shirt|sweater))'
        for clothing in re.findall(clothing_pattern, text, re.IGNORECASE):
            results["suspects"].add(f"wearing {clothing}")

    def _detect_weapons(self, text, results):
        """Multi-method weapon detection"""
        # Method 1: Zero-shot classification
        try:
            classification = self.weapon_classifier(
                text,
                candidate_labels=self.weapon_categories,
                multi_label=True
            )
            for label, score in zip(classification['labels'], classification['scores']):
                if score >= 0.4:  # Lower threshold for police calls
                    results["weapons"].add(label)
        except Exception as e:
            print(f"Weapon classification error: {e}")
        
        # Method 2: Contextual regex pattern
        weapon_pattern = r'(?:has|with|pulled\s+out|brandishing|wielding|shot\s+with)\s+(?:a\s+)?(gun|knife|weapon|firearm|pistol|rifle|handgun)'
        for weapon in re.findall(weapon_pattern, text, re.IGNORECASE):
            results["weapons"].add(weapon.lower())

    def update_weapon_categories(self, new_categories):
        """Dynamically update weapon categories"""
        self.weapon_categories = new_categories

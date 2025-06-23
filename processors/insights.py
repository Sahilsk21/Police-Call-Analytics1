from transformers import pipeline
from config import MODELS
from utils.cache import load_model

class CrimeClassifier:
    def __init__(self):
        # Load zero-shot classification model
        self.classifier = load_model(
            "zero_shot_classifier",
            lambda: pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli")
        )
        
        # Define default crime categories (can be modified dynamically)
        self.categories = [
            "Robbery", 
            "Assault", 
            "Cybercrime", 
            "Burglary", 
            "Theft",
            "Homicide", 
            "Kidnapping", 
            "Arson",
            "Vandalism",
            "Fraud",
            "Drug offense",
            "Other"
        ]
    
    def update_categories(self, new_categories):
        """Update the crime categories dynamically"""
        self.categories = new_categories
        
    def classify(self, text, threshold=0.5, multi_label=False):
        """
        Classify crime text using zero-shot approach
        
        Args:
            text (str): Input text to classify
            threshold (float): Confidence threshold (0-1)
            multi_label (bool): Whether to allow multiple labels
            
        Returns:
            dict: {
                'label': best matching category,
                'confidence': float,
                'scores': dict of all category scores
            }
        """
        result = self.classifier(
            text,
            candidate_labels=self.categories,
            multi_label=multi_label
        )
        
        # Get best match
        best_label = result['labels'][0]
        best_score = result['scores'][0]
        
        # Apply threshold
        if best_score < threshold:
            best_label = "Other"
            
        return {
            'label': best_label,
            'confidence': float(best_score),
            'scores': dict(zip(result['labels'], result['scores']))
        }

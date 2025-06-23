# Police-Call-Analytics1 
Transforming Emergency Response with AI
When a 911 call comes in, every second counts. Police departments face the challenge of quickly extracting crucial details from often chaotic, emotional emergency calls. This system cuts through the noise—literally. 


How It Works: From Panicked Call to Actionable Data
1. Hearing Through the Chaos
Works with real-world recordings: static-filled cell phone calls, background screams, heavy accents

Handles overlapping voices (common in domestic violence calls)

Catches critical details even when callers are whispering

Real Example:

"He's got a—[gunshot]—oh God! 1425 Maple—[unintelligible screaming]—he's wearing a black hoodie!"

The system identifies:
. Gun mention
. Address fragment ("1425 Maple")
. Suspect clothing

2. Understanding Cop Talk
Trained on actual police terminology:

"10-54" → Suspicious person

"Signal 30" → Burglary in progress

"Code 3" → Emergency response needed

Detects subtle cues most systems miss:

"She said she 'tripped down the stairs' but the neighbor reported yelling."
Flags potential domestic violence case despite victim's denial.

3. Building the Suspect Picture
Instead of just noting "white male," it constructs:

Approximate age ("looks about my dad's age" → estimates 40-55)

Clothing details ("Yankees cap, red shoes")

Behavioral cues ("keeps touching his waistband" → possible concealed weapon) 

Getting Started
What You'll Need
Any computer less than 5 years old (even a Raspberry Pi works for testing)

2GB free space for language models

A sample 911 call (we include test recordings if you don't have real ones)

Installation That Won't Frustrate You 

Prerequisites
Python 3.8+

FFmpeg (for audio processing)

Hugging Face API key
# No Docker, no Kubernetes - just Python
git clone https://github.com/sahilsk21/police-call-analytics1.git
cd police-call-analytics1
python -m venv venv
source venv/bin/activate  # or 'venv\Scripts\activate' on Windows
pip install -r requirements.txt  
streamlit run app.py

First Run Tips: 
python analyze.py --audio real_call.wav --sensitivity high  


Technical Components
1. Audio Processing Module
Input: MP3/WAV recordings

Output: Transcribed text (English)

Tech Stack:

Whisper ASR (OpenAI) for speech recognition

Helsinki-NLP for translation (50+ languages)

PyDub for audio format conversion

2. Natural Language Processing (NLP) Engine
Hybrid Extraction:

Regex Patterns (e.g., \d{3,4}\s+[A-Za-z\s]+ for addresses)

Transformer Models (dslim/bert-base-NER)

Zero-Shot Classification (facebook/bart-large-mnli) 

3. Crime Classification Module
Supported Crime Types:

Category	Examples
Violent Crimes	
Assault, Homicide
Property Crimes	 
Burglary, Theft
Special Cases	
Kidnapping, Cybercrime
Model: facebook/bart-large-mnli (85%+ accuracy)

4. User Interface
Streamlit Dashboard

Upload audio files

View extracted entities

Export reports (JSON/PDF)

Sample Output: 

{
  "incident_type": {"label": "Armed Robbery", "confidence": 0.87},
  "locations": ["2350 Main St", "First National Bank"],
  "suspects": ["white male ~35", "wearing red hoodie"],
  "weapons": ["handgun"],
  "timeline": {
    "call_received": "14:30",
    "incident_occurred": "approximately 14:25" 
  }
}
import streamlit as st
from transformers import pipeline
import tempfile
import os
from pydub import AudioSegment
import torch
from datetime import datetime
import json

# App Configuration
st.set_page_config(
    page_title="Police Call Analytics",
    
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "whisper": pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device
        ),
        "translator": pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-mul-en",
            device=device
        ),
        "ner": pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=device
        )
    }

def process_audio(uploaded_file):
    try:
        models = load_models()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            if uploaded_file.name.endswith(".mp3"):
                audio = AudioSegment.from_mp3(uploaded_file)
                audio.export(tmp.name, format="wav")
            else:
                tmp.write(uploaded_file.getvalue())
            
            result = models["whisper"](tmp.name)
            transcript = result["text"]
            
            is_english = all(ord(c) < 128 for c in transcript)
            if not is_english:
                translated = models["translator"](transcript)[0]["translation_text"]
                translation = True
            else:
                translated = transcript
                translation = False
            
            entities = models["ner"](translated)
            
            entity_data = {
                "locations": set(),
                "times": set(),
                "weapons": set(),
                "suspects": set()
            }
            
            for entity in entities:
                group = entity["entity_group"]
                word = entity["word"]
                
                if group in ["LOC", "GPE"]:
                    entity_data["locations"].add(word)
                elif group == "PER":
                    entity_data["suspects"].add(word)
                elif group in ["DATE", "TIME"]:
                    entity_data["times"].add(word)
            
            weapon_words = {'gun', 'knife', 'weapon', 'pistol'}
            for word in translated.lower().split():
                if word in weapon_words:
                    entity_data["weapons"].add(word)
            
            return {
                "metadata": {
                    "filename": uploaded_file.name,
                    "processed_at": datetime.now().isoformat(),
                    "file_size": f"{uploaded_file.size/1024:.1f} KB",
                    "language": "en" if is_english else "non-en"
                },
                "transcript": {
                    "original": transcript,
                    "translated": translated,
                    "was_translated": translation
                },
                "classification": {
                    "category":[ "Assault","Robbery","Cybercrime","Burglary","Vandalism","Harassment","Fraud","Kidnapping","Arson","DrugOffense","Other"],  
                    "confidence": 0.3      
                },
                "entities": {k: list(v) for k, v in entity_data.items()}
            }
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None
    finally:
        if 'tmp' in locals() and os.path.exists(tmp.name):
            os.unlink(tmp.name)

def display_results(results):
    st.header("Analysis Results")
    
    if results['transcript']['was_translated']:
        st.warning("Text was automatically translated from original audio")
    
    conf = results['classification']['confidence']
    st.metric(
        "Crime Classification",
        results['classification']['category'],
        delta=f"{conf:.0%} confidence",
        delta_color="normal" if conf > 0.7 else "inverse"
    )
    
    # Create three tabs instead of two
    tab1, tab2, tab3 = st.tabs(["Entities", "Transcript", "Raw Data"])
    
    with tab1:
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Locations")
            st.write("\n".join(f"- {loc}" for loc in results['entities']['locations']) or "None found")
            
            st.subheader("Time References")
            st.write("\n".join(f"- {time}" for time in results['entities']['times']) or "None found")
        
        with cols[1]:
            st.subheader("Weapons")
            st.write("\n".join(f"- {weapon}" for weapon in results['entities']['weapons']) or "None found")
            
            st.subheader("Suspects")
            st.write("\n".join(f"- {suspect}" for suspect in results['entities']['suspects']) or "None found")
    
    with tab2:
        st.subheader("Transcript")
        st.text_area("", results['transcript']['translated'], height=300)
        
        if results['transcript']['was_translated']:
            with st.expander("View Original Text"):
                st.text(results['transcript']['original'])
    
    with tab3:
        st.subheader("Complete Raw Data")
        st.json(results)
        
        st.download_button(
            "Download Full Report",
            json.dumps(results, indent=2),
            file_name=f"police_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def main():
    st.title("Police Call Analytics")
    st.markdown("AI-powered analysis of police call recordings")
    
    uploaded_file = st.file_uploader(
        "Upload recording (MP3/WAV)",
        type=["mp3", "wav"]
    )
    
    if uploaded_file and st.button("Analyze"):
        with st.spinner("Processing... (This may take 1-2 minutes)"):
            results = process_audio(uploaded_file)
            if results:
                st.session_state.results = results
                st.success("Analysis complete!")
                st.balloons()
    
    if st.session_state.results:
        display_results(st.session_state.results)

if __name__ == "__main__":
    main()

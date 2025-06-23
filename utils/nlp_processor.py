from functools import lru_cache
import re
import warnings

try:
    from transformers import pipeline
except ImportError:
    raise ImportError("Install required packages: pip install transformers")

@lru_cache(maxsize=2)
def get_ner_pipeline(language='en'):
    model_map = {
        'en': "dslim/bert-base-NER",
        'other': "Davlan/bert-base-multilingual-cased-ner-hrl"
    }
    model_name = model_map.get(language, model_map['other'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple"
            )
    except Exception as e:
        raise RuntimeError(f"NER model load failed: {str(e)}")

def extract_entities(text, language='en'):
    if not text.strip():
        return empty_entity_response()

    try:
        ner = get_ner_pipeline(language)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entities = ner(text)
        return process_entities(entities, text, language)
    except Exception as e:
        print(f"Entity extraction failed: {str(e)}")
        return empty_entity_response()

def empty_entity_response():
    return {
        "locations": [],
        "times": [],
        "weapons": [],
        "suspects": [],
        "organizations": []
    }

def process_entities(entities, text, language):
    entity_map = {
        'LOC': "locations",
        'GPE': "locations",
        'DATE': "times",
        'TIME': "times",
        'PER': "suspects",
        'ORG': "organizations"
    }

    results = {v: set() for v in entity_map.values()}
    for entity in entities:
        group = entity.get('entity_group')
        if group in entity_map:
            results[entity_map[group]].add(entity['word'])

    results = {k: list(v) for k, v in results.items()}
    results["weapons"] = detect_weapons(text, language)
    return results

def detect_weapons(text, language='en'):
    patterns = {
        'en': [
            r'\b(knife|knives|blade|razor)\b',
            r'\b(gun|firearm|pistol|revolver|rifle|shotgun)\b',
            r'\b(bat|club|hammer|crowbar)\b',
            r'\b(explosive|bomb|grenade)\b'
        ],
        'es': [r'\b(cuchillo|navaja|pistola|rev\u00f3lver)\b'],
        'fr': [r'\b(couteau|pistolet|revolver)\b']
    }

    lang_patterns = patterns.get(language, patterns['en'])
    weapons = set()

    for pattern in lang_patterns:
        weapons.update(re.findall(pattern, text, re.IGNORECASE))

    return list(weapons)


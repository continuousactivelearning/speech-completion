from transformers import pipeline
import spacy

# Load models once
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
nlp_syntax = spacy.load("en_core_web_sm")

def extract_triplets(doc):
    """Extract basic (subject, verb, object) relations using spaCy."""
    triplets = set()
    for sent in doc.sents:
        subj = verb = obj = None
        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            elif "obj" in token.dep_:
                obj = token.text
            elif token.pos_ == "VERB":
                verb = token.lemma_
        if subj and verb and obj:
            triplets.add((subj.lower(), verb.lower(), obj.lower()))
    return triplets

def count_kg_elements(text):
    """Returns the number of unique entities + relations in a given text."""
    
    # Named Entities (via HF)
    ner_results = ner_pipeline(text)
    unique_entities = set(ent['word'].strip().lower() for ent in ner_results)

    # Relations (via spaCy)
    doc = nlp_syntax(text)
    unique_relations = extract_triplets(doc)

    total_kg_elements = len(unique_entities) + len(unique_relations)
    return total_kg_elements
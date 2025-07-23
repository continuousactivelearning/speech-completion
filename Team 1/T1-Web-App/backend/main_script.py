from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import spacy
import random
from completion_model import load_glove, load_bigru_model, predict_bigru_completion


# For approach 1 
glove_vectors = load_glove("glove.6B.300d.txt")
bigru_model = load_bigru_model("models/BiGRUTextNet_GloVe.pt")


#For approach 2 LLM
from FineTuned_Gemma2B import generate_from_base
from summariser import extractive_summarize
from keyword_generator import extract_deduplicated_keywords
from knowledge_graph import count_kg_elements
from Is_Unique_Keyword import is_unique

app = Flask(__name__)
CORS(app)

kw_model = KeyBERT()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
nlp_syntax = spacy.load("en_core_web_sm")

# --- Global State ---
state = {
    "chunks": [],
    "unique_keywords": [],
    "unique_keyword_embeddings": [],
    "random_values": [],
    "entity_mention_cumsum": [],
    "convergence_chunk": None,
    "kdrs": [],
    "cum_keywords": [],
    "kdr_sum_window": [],
    "global_keyword_list": [],
    "global_content_generated": "",
    "first_four_percentages": [],
    "cumulative_text": "",
    "last_returned_pct": 0.0,
    "count_kg_real": 0
}

KDR_THRESHOLD = 0.02
KDR_WINDOW_LIMIT = 3

# --- Check Keyword Uniqueness via Embedding ---
def is_semantically_unique(keyword, existing_embeddings, threshold=0.8):
    keyword_emb = embedding_model.encode([keyword])
    if not existing_embeddings:
        return True
    sims = cosine_similarity(keyword_emb, existing_embeddings)
    return sims.max() < threshold

# --- Reset State ---
@app.route('/reset', methods=['POST'])
def reset():
    state.clear()
    state.update({
        "chunks": [],
        "unique_keywords": [],
        "unique_keyword_embeddings": [],
        "random_values": [],
        "entity_mention_cumsum": [],
        "convergence_chunk": None,
        "kdrs": [],
        "cum_keywords": [],
        "kdr_sum_window": [],
        "global_keyword_list": [],
        "global_content_generated": "",
        "first_four_percentages": [],
        "cumulative_text": "",
        "last_returned_pct": 0.0,
        "count_kg_real": 0
    })
    return {"status": "reset"}

# --- Analyze Route ---
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    chunk = data.get("chunk", "")
    chunk_number = len(state['chunks']) + 1

    if chunk.strip():  # only if chunk has real content
        state["cumulative_text"] += " " + chunk

    print(f"\nReceived chunk #{chunk_number} with {len(chunk.split())} words")

    # --- Keyword Discovery ---
    keywords = kw_model.extract_keywords(chunk, top_n=15, stop_words='english')
    new_keywords = 0
    for kw, _ in keywords:
        if is_semantically_unique(kw, state['unique_keyword_embeddings']):
            state['unique_keywords'].append(kw)
            state['unique_keyword_embeddings'].append(embedding_model.encode(kw))
            new_keywords += 1

    cumulative_keywords = len(state['unique_keywords'])
    kdr = new_keywords / cumulative_keywords if cumulative_keywords != 0 else 0

    state['chunks'].append(chunk_number)
    state['kdrs'].append(kdr)
    state['cum_keywords'].append(cumulative_keywords)

    

    # --- Entity Count ---
    entities = ner_pipeline(chunk)
    ner_count = len(entities)
    prev_ner_cum = state['entity_mention_cumsum'][-1] if state['entity_mention_cumsum'] else 0
    state['entity_mention_cumsum'].append(prev_ner_cum + ner_count)



    print(f"Extracted {len(keywords)} keywords, New unique: {new_keywords}, Total: {cumulative_keywords}")
    print(f"NER Count: {ner_count}, Cumulative NER: {state['entity_mention_cumsum'][-1]}")



    # --- KDR Rolling Convergence ---
    if kdr < KDR_THRESHOLD:
        state["kdr_sum_window"].append(chunk_number)
    if len(state["kdr_sum_window"]) >= KDR_WINDOW_LIMIT and state["convergence_chunk"] is None:
        state["convergence_chunk"] = state["kdr_sum_window"][-1]

    # --- Concatenate all chunks till now (including current one)
    bigru_pct = predict_bigru_completion(state["cumulative_text"].strip(), bigru_model, glove_vectors) / 100.0



    # --- Percentage Completion Calculation ---
    temp_keywords = extract_deduplicated_keywords(chunk, ngram_range=2)
    new_keywords_list = []
    for kw in temp_keywords.split(','):
        if is_unique(kw, state['global_keyword_list']):
            state['global_keyword_list'].append(kw)
            new_keywords_list.append(kw)

    summarized = extractive_summarize(chunk)
    real_kg = count_kg_elements(chunk) * 0.5
    state['count_kg_real'] += real_kg

    if chunk_number > 5:
        if len(new_keywords_list) > 6:
            gen = generate_from_base("generate standard content with the following keywords:", 800, ",".join(new_keywords_list))
        elif 2 < len(new_keywords_list) <= 6:
            gen = generate_from_base("generate standard content with the following keywords:", 110 * len(new_keywords_list), ",".join(new_keywords_list))
        else:
            gen = generate_from_base("generate standard content with the following keywords:", 200, ",".join(new_keywords_list))
    else:
        gen = generate_from_base("generate standard content with the following keywords:", 1560, ",".join(new_keywords_list))

    state['global_content_generated'] += gen

    if chunk_number > 4:
        try:
            percentage_completion = state['count_kg_real'] / count_kg_elements(state['global_content_generated'])
        except ZeroDivisionError:
            percentage_completion = -1
        print(f" Percentage Completion: {percentage_completion:.2f}")
    else:
        print(" Early phase. Not enough data yet.")
        try:
            partial_pct = state['count_kg_real'] / count_kg_elements(state['global_content_generated'])
            state['first_four_percentages'].append(partial_pct)
        except ZeroDivisionError:
            state['first_four_percentages'].append(0.0)
        percentage_completion = -1
    
    # --- Weighted Average ---    


    if percentage_completion!=-1:
        final_completion_pct = 0.5 * percentage_completion + 0.5 * bigru_pct
    else:
        final_completion_pct = -1

    if chunk_number > 6 and final_completion_pct != -1:
        if final_completion_pct < state.get("last_returned_pct", 0.0):
            final_completion_pct = state["last_returned_pct"] + 0.5

    if final_completion_pct != -1:
        state["last_returned_pct"] = final_completion_pct

    # --- Return Response ---
    return jsonify({
        "chunk": chunk_number,
        "cumulative_keywords": cumulative_keywords,
        "kdr": kdr,
        "random_value": final_completion_pct * 100 if final_completion_pct != -1 else -1,  # writ random no for testing
        "entity_relation_discovery": state['entity_mention_cumsum'][-1],
        "convergence_chunk": state['convergence_chunk']
    })

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
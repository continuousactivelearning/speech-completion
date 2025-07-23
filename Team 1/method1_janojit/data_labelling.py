import os
import re
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

# Load models
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

def load_and_segment_transcript(txt_file_path):
    segmented_sentences = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        full_combined_text = []
        for line in lines:
            line = line.strip()
            if line:
                match = re.match(r'^\s*(.*)', line)
                if match:
                    full_combined_text.append(match.group(1).strip())

        combined_text = " ".join(full_combined_text)
        sentences = re.split(r'(?<!Mr|Ms|Dr|St|Vs)\.\s+|\!\s+|\?\s+', combined_text)

        for sentence in sentences:
            s = sentence.strip()
            if s:
                segmented_sentences.append(s)

    except Exception as e:
        print(f"Error loading or parsing transcript {txt_file_path}: {e}")

    return segmented_sentences

def semantic_completion(sentences):
    embeddings = sentence_model.encode(sentences)
    info_gains = []

    for i in range(2, len(embeddings)):
        X_train = embeddings[:i-1]
        y_train = embeddings[1:i]

        model_reg = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation='logistic',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        model_reg.fit(X_train, y_train)
        predicted = model_reg.predict([embeddings[i-1]])[0]
        actual = embeddings[i]

        sim_pred = cosine_similarity([predicted], [actual])[0][0]
        gain_pred = 1 - sim_pred
        sim_redundancy = cosine_similarity([actual], embeddings[:i]).max()
        gain_redundancy = 1 - sim_redundancy

        combined_info_gain = min(gain_pred, gain_redundancy)
        info_gains.append(combined_info_gain)

    info_gains = [0.01, 0.01] + info_gains
    y_completion_percentage = np.cumsum(info_gains) / np.sum(info_gains)
    return list(y_completion_percentage * 100)

def extract_entities_and_relations(text, nlp_model):
    doc = nlp_model(text)
    entities = {(ent.text.strip(), ent.label_) for ent in doc.ents}

    relations = set()
    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ == "ROOT":
            subjects = [w for w in tok.children if "subj" in w.dep_]
            objects = [w for w in tok.children if "obj" in w.dep_]
            for subj in subjects:
                for obj in objects:
                    relations.add((subj.text.strip(), tok.lemma_.strip(), obj.text.strip()))
    return entities, relations

def kg_completion(sentences, nlp_model):
    entities_total = set()
    relations_total = set()

    for sent in sentences:
        e, r = extract_entities_and_relations(sent, nlp_model)
        entities_total.update(e)
        relations_total.update(r)

    num_total_kg_elements = len(entities_total) + len(relations_total)

    entities_seen = set()
    relations_seen = set()
    kg_completion_percentages = []

    for sent in sentences:
        e, r = extract_entities_and_relations(sent, nlp_model)
        entities_seen.update(e)
        relations_seen.update(r)
        seen = len(entities_seen) + len(relations_seen)
        percent = (seen / num_total_kg_elements) * 100 if num_total_kg_elements > 0 else 0
        kg_completion_percentages.append(percent)

    return kg_completion_percentages

def compute_combined_completion_for_file(file_path):
    sentences = load_and_segment_transcript(file_path)
    if not sentences:
        return pd.DataFrame()

    method1 = semantic_completion(sentences)
    method2 = kg_completion(sentences, nlp)

    video_name = os.path.basename(file_path).rsplit(".", 1)[0]

    # Create cumulative sentence concatenation and final DataFrame
    cumulative_sentences = []
    full_text = ""
    for sent in sentences:
        if full_text:
            full_text += ". " + sent
        else:
            full_text = sent
        cumulative_sentences.append(full_text.strip())

    final_df = pd.DataFrame({
        'Sentence': cumulative_sentences,
        'video_name': video_name,
        'completion_percentage_method1': method1,
        'completion_percentage_method2': method2
    })

    final_df['completion_percentage'] = final_df[['completion_percentage_method1', 'completion_percentage_method2']].mean(axis=1)
    return final_df

# ---- Loop through all .txt files in a folder ----

def compute_combined_completion_for_folder(folder_path):
    all_dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            df = compute_combined_completion_for_file(file_path)
            if not df.empty:
                all_dataframes.append(df)

    final_df = pd.concat(all_dataframes, ignore_index=True)
    return final_df


folder_path = "./transcripts"
final_df = compute_combined_completion_for_folder(folder_path)
final_df.to_csv("combined_completion_all_videos.csv", index=False)
print("\nSaved all video data to CSV.")
print(final_df.head())
from keybert import KeyBERT


kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")  
def extract_deduplicated_keywords(transcript: str, ngram_range: int = 1, top_n: int = 10) -> str:
    extracted = kw_model.extract_keywords(
        transcript,
        keyphrase_ngram_range=(1, ngram_range),
        stop_words='english',
        top_n=top_n
    )

    keywords_dict = {kw: round(score, 4) for kw, score in extracted}

    final_keywords = {}
    for phrase, score in sorted(keywords_dict.items(), key=lambda x: -x[1]):
        if not any(word in final_keywords for word in phrase.split()):
            final_keywords[phrase] = score

    return ', '.join(final_keywords.keys())

import pickle, json, numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# 리소스 로드
with open("data/bm25_genre.pkl", "rb") as f:
    BM25_GENRE = pickle.load(f)
with open("data/bm25_artist.pkl", "rb") as f:
    BM25_ARTIST = pickle.load(f)
EMBEDS = np.load("data/embeddings.npy")
with open("data/doc_ids.pkl", "rb") as f:
    DOC_IDS = pickle.load(f)
with open("data/concerts.json", "r", encoding="utf-8") as f:
    CONCERTS_META = {str(c["id"]): c for c in json.load(f)}

ID2IDX = {str(cid): i for i, cid in enumerate(DOC_IDS)}

def recommend(user_genres, user_artists, user_fav_ids, topk=10):
    genre_scores = BM25_GENRE.get_scores(user_genres or [])
    artist_scores = BM25_ARTIST.get_scores(user_artists or [])

    fav_idx = [ID2IDX[fid] for fid in user_fav_ids if fid in ID2IDX]
    if fav_idx:
        user_vec = EMBEDS[fav_idx].mean(axis=0, keepdims=True)
        sim_scores = cosine_similarity(user_vec, EMBEDS)[0]
    else:
        sim_scores = np.zeros(len(DOC_IDS), dtype=np.float32)

    final_scores = 0.2 * genre_scores + 0.2 * artist_scores + 0.6 * sim_scores

    sorted_indices = np.argsort(-final_scores)

    fav_set = set(user_fav_ids)
    filtered_indices = [idx for idx in sorted_indices if str(DOC_IDS[idx]) not in fav_set]

    top_indices = filtered_indices[:topk]

    results = []
    for idx in top_indices:
        cid = DOC_IDS[idx]
        meta = CONCERTS_META.get(str(cid), {"name": "Unknown"})
        results.append({
            "id": str(cid),
            "name": meta.get("name", "Unknown"),
            "score": float(final_scores[idx])
        })
    return results
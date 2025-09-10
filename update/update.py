import os, pickle, json
import numpy as np
from rank_bm25 import BM25Okapi
from models.embedder import encode_keywords

DATA_DIR = "data"

CONCERT_JSON = os.path.join(DATA_DIR, "concerts.json")
BM25_GENRE_PKL = os.path.join(DATA_DIR, "bm25_genre.pkl")
BM25_ARTIST_PKL = os.path.join(DATA_DIR, "bm25_artist.pkl")
EMBEDDINGS_NPY = os.path.join(DATA_DIR, "embeddings.npy")
DOC_IDS_PKL = os.path.join(DATA_DIR, "doc_ids.pkl")

# 장르 매핑
GENRE_MAP = {
    1: "케이팝",
    2: "락/메탈",
    3: "발라드",
    4: "랩/힙합",
    5: "팬미팅",
    6: "인디",
    7: "재즈/소울",
    8: "내한공연",
    9: "R&B",
    10: "EDM",
    11: "디너쇼",
    12: "기타"
}

# 기존 파일 로드
def load_existing_data():
    if os.path.exists(CONCERT_JSON):
        with open(CONCERT_JSON, "r", encoding="utf-8") as f:
            concerts = json.load(f)
    else:
        concerts = []

    if os.path.exists(BM25_GENRE_PKL):
        with open(BM25_GENRE_PKL, "rb") as f:
            bm25_genre = pickle.load(f)
    else:
        bm25_genre = None

    if os.path.exists(BM25_ARTIST_PKL):
        with open(BM25_ARTIST_PKL, "rb") as f:
            bm25_artist = pickle.load(f)
    else:
        bm25_artist = None

    if os.path.exists(EMBEDDINGS_NPY):
        embeddings = np.load(EMBEDDINGS_NPY)
    else:
        embeddings = np.zeros((0, 384))

    if os.path.exists(DOC_IDS_PKL):
        with open(DOC_IDS_PKL, "rb") as f:
            doc_ids = pickle.load(f)
    else:
        doc_ids = []

    return concerts, bm25_genre, bm25_artist, embeddings, doc_ids

# BM25 빌드
def build_bm25(corpus):
    return BM25Okapi(corpus)

# ---------------- gRPC 호출 시 업데이트 함수 ----------------
def update_concerts(concerts_to_add):
    if not concerts_to_add:
        print("업데이트 공연 없음")
        return

    print(f"업데이트/추가 공연 수: {len(concerts_to_add)}")

    concerts_existing, bm25_genre, bm25_artist, embeddings, doc_ids = load_existing_data()

    # JSON 백업
    existing_ids = set(c["id"] for c in concerts_existing)
    concerts_existing.extend([c for c in concerts_to_add if c["id"] not in existing_ids])
    with open(CONCERT_JSON, "w", encoding="utf-8") as f:
        json.dump(concerts_existing, f, ensure_ascii=False, indent=2)

    # BM25 인덱스 업데이트
    genre_corpus = [[GENRE_MAP.get(c["genre"], "기타")] for c in concerts_existing]
    artist_corpus = [c["cast"] for c in concerts_existing]

    bm25_genre = build_bm25(genre_corpus)
    bm25_artist = build_bm25(artist_corpus)

    with open(BM25_GENRE_PKL, "wb") as f:
        pickle.dump(bm25_genre, f)
    with open(BM25_ARTIST_PKL, "wb") as f:
        pickle.dump(bm25_artist, f)

    # 키워드 임베딩 업데이트
    keywords_list = [c["keyword"] for c in concerts_to_add]
    new_embeddings = np.stack([encode_keywords(k) for k in keywords_list])
    embeddings = np.vstack([embeddings, new_embeddings]) if embeddings.size > 0 else new_embeddings

    # doc_ids 업데이트
    doc_ids.extend([c["id"] for c in concerts_to_add if c["id"] not in doc_ids])

    np.save(EMBEDDINGS_NPY, embeddings)
    with open(DOC_IDS_PKL, "wb") as f:
        pickle.dump(doc_ids, f)

    print("BM25 인덱스, 임베딩, JSON 백업 완료")
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
    import os, json, pickle, numpy as np

    print("🔍 [DEBUG] load_existing_data() 호출됨")

    # 🎵 공연 JSON 로드
    if os.path.exists(CONCERT_JSON):
        print(f"📂 JSON 파일 존재: {CONCERT_JSON}")
        try:
            with open(CONCERT_JSON, "r", encoding="utf-8") as f:
                concerts = json.load(f)
            print(f"✅ 공연 데이터 로드 완료 — 개수: {len(concerts)}")
        except Exception as e:
            print(f"❌ 공연 JSON 로드 실패: {e}")
            concerts = []
    else:
        print(f"⚠️ 공연 JSON 파일 없음: {CONCERT_JSON}")
        concerts = []

    # 🧩 BM25 장르 인덱스 로드
    if os.path.exists(BM25_GENRE_PKL):
        print(f"📂 BM25 장르 인덱스 존재: {BM25_GENRE_PKL}")
        try:
            with open(BM25_GENRE_PKL, "rb") as f:
                bm25_genre = pickle.load(f)
            print("✅ BM25 장르 인덱스 로드 완료")
        except Exception as e:
            print(f"❌ BM25 장르 로드 실패: {e}")
            bm25_genre = None
    else:
        print(f"⚠️ BM25 장르 인덱스 없음: {BM25_GENRE_PKL}")
        bm25_genre = None

    # 🎤 BM25 아티스트 인덱스 로드
    if os.path.exists(BM25_ARTIST_PKL):
        print(f"📂 BM25 아티스트 인덱스 존재: {BM25_ARTIST_PKL}")
        try:
            with open(BM25_ARTIST_PKL, "rb") as f:
                bm25_artist = pickle.load(f)
            print("✅ BM25 아티스트 인덱스 로드 완료")
        except Exception as e:
            print(f"❌ BM25 아티스트 로드 실패: {e}")
            bm25_artist = None
    else:
        print(f"⚠️ BM25 아티스트 인덱스 없음: {BM25_ARTIST_PKL}")
        bm25_artist = None

    # 🧠 임베딩 로드
    if os.path.exists(EMBEDDINGS_NPY):
        print(f"📂 임베딩 파일 존재: {EMBEDDINGS_NPY}")
        try:
            embeddings = np.load(EMBEDDINGS_NPY)
            print(f"✅ 임베딩 로드 완료 — shape: {embeddings.shape}")
        except Exception as e:
            print(f"❌ 임베딩 로드 실패: {e}")
            embeddings = np.zeros((0, 384))
    else:
        print(f"⚠️ 임베딩 파일 없음: {EMBEDDINGS_NPY}")
        embeddings = np.zeros((0, 384))

    if os.path.exists(DOC_IDS_PKL):
        print(f"📂 문서 ID 파일 존재: {DOC_IDS_PKL}")
        try:
            with open(DOC_IDS_PKL, "rb") as f:
                doc_ids = pickle.load(f)
            print(f"✅ 문서 ID 로드 완료 — 개수: {len(doc_ids)}")
        except Exception as e:
            print(f"❌ 문서 ID 로드 실패: {e}")
            doc_ids = []
    else:
        print(f"⚠️ 문서 ID 파일 없음: {DOC_IDS_PKL}")
        doc_ids = []

    print("🎯 [DEBUG] load_existing_data() 완료")
    return concerts, bm25_genre, bm25_artist, embeddings, doc_ids

# BM25 빌드
def build_bm25(corpus):
    return BM25Okapi(corpus)

# ---------------- gRPC 호출 시 업데이트 함수 ----------------
def update_concerts(concerts_to_add):
    import json, pickle, numpy as np

    print("🔍 [DEBUG] update_concerts() 호출됨")

    if not concerts_to_add:
        print("ℹ️  업데이트할 공연 없음")
        return

    print(f"📦 업데이트/추가 공연 수: {len(concerts_to_add)}")

    # 기존 데이터 로드
    print("📂 기존 데이터 로드 중...")
    concerts_existing, bm25_genre, bm25_artist, embeddings, doc_ids = load_existing_data()
    print(f"✅ 기존 공연 수: {len(concerts_existing)}")
    print(f"✅ 기존 임베딩 shape: {embeddings.shape if embeddings.size > 0 else '(empty)'}")
    print(f"✅ 기존 문서 ID 수: {len(doc_ids)}")

    # JSON 백업
    print("📝 JSON 파일 백업 중...")
    existing_ids = set(c["id"] for c in concerts_existing)
    before_count = len(concerts_existing)
    new_items = [c for c in concerts_to_add if c["id"] not in existing_ids]
    concerts_existing.extend(new_items)
    after_count = len(concerts_existing)
    print(f"➡️  신규 추가된 공연 수: {len(new_items)} (총 {before_count} → {after_count})")

    try:
        with open(CONCERT_JSON, "w", encoding="utf-8") as f:
            json.dump(concerts_existing, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON 저장 완료 → {CONCERT_JSON}")
    except Exception as e:
        print(f"❌ JSON 저장 중 오류 발생: {e}")

    # BM25 인덱스 업데이트
    print("⚙️ BM25 인덱스 생성 중...")
    genre_corpus = [[GENRE_MAP.get(c["genre"], "기타")] for c in concerts_existing]
    artist_corpus = [c["cast"] for c in concerts_existing]
    print(f"🧩 장르 코퍼스 크기: {len(genre_corpus)}, 아티스트 코퍼스 크기: {len(artist_corpus)}")

    bm25_genre = build_bm25(genre_corpus)
    bm25_artist = build_bm25(artist_corpus)

    try:
        with open(BM25_GENRE_PKL, "wb") as f:
            pickle.dump(bm25_genre, f)
        with open(BM25_ARTIST_PKL, "wb") as f:
            pickle.dump(bm25_artist, f)
        print(f"✅ BM25 인덱스 저장 완료 → {BM25_GENRE_PKL}, {BM25_ARTIST_PKL}")
    except Exception as e:
        print(f"❌ BM25 저장 중 오류 발생: {e}")

    # 키워드 임베딩 업데이트
    print("🧠 키워드 임베딩 업데이트 중...")
    keywords_list = [c["keyword"] for c in concerts_to_add]
    print(f"➡️ 신규 키워드 수: {len(keywords_list)}")

    try:
        new_embeddings = np.stack([encode_keywords(k) for k in keywords_list])
        embeddings = np.vstack([embeddings, new_embeddings]) if embeddings.size > 0 else new_embeddings
        print(f"✅ 임베딩 shape 업데이트 완료: {embeddings.shape}")
    except Exception as e:
        print(f"❌ 임베딩 계산 중 오류 발생: {e}")
        return

    # doc_ids 업데이트
    print("🆔 문서 ID 리스트 업데이트 중...")
    new_doc_ids = [c["id"] for c in concerts_to_add if c["id"] not in doc_ids]
    before_len = len(doc_ids)
    doc_ids.extend(new_doc_ids)
    print(f"➡️ 신규 ID 수: {len(new_doc_ids)} (총 {before_len} → {len(doc_ids)})")

    try:
        np.save(EMBEDDINGS_NPY, embeddings)
        with open(DOC_IDS_PKL, "wb") as f:
            pickle.dump(doc_ids, f)
        print(f"✅ 임베딩 및 문서 ID 저장 완료 → {EMBEDDINGS_NPY}, {DOC_IDS_PKL}")
    except Exception as e:
        print(f"❌ 저장 중 오류 발생: {e}")

    print("🎉 BM25 인덱스, 임베딩, JSON 백업 완료")

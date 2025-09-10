from sentence_transformers import SentenceTransformer
import numpy as np

# 모델 로드
_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_keywords(keywords) -> np.ndarray:
    """
    공연 키워드 20개 -> 평균 pooling -> 1개 벡터 반환
    keywords: list[str] 또는 "|" 구분된 str
    """
    if not keywords:
        return np.zeros(_model.get_sentence_embedding_dimension())

    if isinstance(keywords, str):
        keywords = keywords.split("|")

    embs = _model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)

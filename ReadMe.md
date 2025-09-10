data/
├─ concerts.json          # 전체 공연 메타데이터 백업
├─ bm25_genre.pkl         # 장르 BM25 인덱스
├─ bm25_artist.pkl        # 아티스트 BM25 인덱스
├─ embeddings.npy         # 공연 키워드 임베딩 벡터
├─ doc_ids.pkl            # 공연 ID 리스트 (임베딩/문서 매핑용)

update/
├─ update_server.py       # gRPC 서버: 백엔드 호출 시 새 공연 데이터 증분 업데이트
├─ update.proto           # gRPC 정의 파일
├─ update.py              # 업데이트 로직 (BM25/임베딩/JSON 갱신)

recommendation/
├─ recommender_server.py  # gRPC 서버: 백엔드 요청 시 추천 공연 반환
├─ recommender.proto      # gRPC 정의 파일
├─ recommender.py         # 추천 알고리즘 로직 (BM25 + 임베딩 점수 계산)
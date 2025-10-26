import grpc
from concurrent import futures
from . import recommender_pb2, recommender_pb2_grpc
from .recommender import recommend  # 추천 로직만 import

class RecommenderServicer(recommender_pb2_grpc.RecommenderServicer):
    def Recommend(self, request, context):
        print("request is accepted")
        user_genres = list(request.genres)
        user_artists = list(request.artists)
        user_fav_ids = list(request.fav_ids)
        topk = request.topk if request.topk else 10

        recs = recommend(user_genres, user_artists, user_fav_ids, topk)
        # gRPC 메시지 변환
        results = [recommender_pb2.Concert(id=r["id"], name=r["name"], score=r["score"]) for r in recs]
        print(results)
        print("done")
        return recommender_pb2.RecommendResponse(concerts=results)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommender_pb2_grpc.add_RecommenderServicer_to_server(RecommenderServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Recommender gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
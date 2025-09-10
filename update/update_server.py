import grpc
from concurrent import futures
import update_pb2, update_pb2_grpc
from update import update_concerts  

class ConcertUpdaterServicer(update_pb2_grpc.ConcertUpdaterServicer):
    def UpdateConcerts(self, request, context):
        concerts_to_add = []
        for c in request.concerts:
            concerts_to_add.append({
                "id": c.id,
                "title": c.title,
                "genre": c.genre,
                "cast": list(c.cast),
                "keyword": list(c.keyword),
                "updated_at": c.updated_at
            })
        update_concerts(concerts_to_add)
        return update_pb2.UpdateResponse(success=True, message="Update completed")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    update_pb2_grpc.add_ConcertUpdaterServicer_to_server(ConcertUpdaterServicer(), server)
    server.add_insecure_port("[::]:50052")  # 포트는 필요에 맞게 변경
    server.start()
    print("ConcertUpdater gRPC server started on port 50052")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

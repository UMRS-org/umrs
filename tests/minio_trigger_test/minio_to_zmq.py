import zmq
import json
from flask import Flask, request

# context = zmq.Context()
# socket = context.socket(zmq.PUB)
# socket.connect("tcp://127.0.0.1:5555")


# ZeroMQ Context & Publisher
# context = zmq.Context()
# socket = context.socket(zmq.PUB)
# socket.bind("tcp://127.0.0.1:5555")  # ZeroMQ Publisher on port 5555


app = Flask(__name__)


@app.route("/minio-events", methods=["POST"])
def minio_events():
    """Receive MinIO Webhook Event and Push to ZeroMQ"""
    data = request.json  # Get JSON data from MinIO
    if data and "Records" in data:
        for record in data["Records"]:
            bucket_name = record["s3"]["bucket"]["name"]
            object_path = record["s3"]["object"]["key"]
            msg = f"s3://{bucket_name}/{object_path}"
            print(f"Received: {msg}")
    return "OK", 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)

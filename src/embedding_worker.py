#!/usr/bin/python3

import argparse
import os
import sys

import zmq

# Ensure src/ is on sys.path when running as a standalone script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from umrs.s3 import load_data_from_s3, connect_to_s3
from umrs.models_loading import load_model
from umrs.utils import zero_out_embeddings_for_empty_objects


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UMRS Embedding Worker")
    parser.add_argument(
        "--bind",
        default=os.getenv("EMBEDDING_WORKER_BIND", "tcp://0.0.0.0:6000"),
        help="ZeroMQ bind address for the worker REP socket",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("EMBEDDING_DEVICE", None),
        help="Optional device hint (e.g. 'cuda' or 'cpu'), passed to model loader if applicable",
    )
    parser.add_argument(
        "--s3-config",
        default=os.getenv("EMBEDDING_S3_CONFIG", os.path.expanduser("~/.s3cfg")),
        help="Path to S3 configuration file used by the worker",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[worker] Starting UMRS embedding worker on {args.bind}")

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)

    s3_cfg_path = os.path.expanduser(args.s3_config)
    print(f"[worker] Using S3 config: {s3_cfg_path}")
    s3_connection, bucket_location = connect_to_s3(s3_cfg_path)

    # Simple in-memory model cache: (modality_name, model_name, device) -> model
    model_cache = {}

    while True:
        request = socket.recv_json()
        job_id = request.get("job_id")

        try:
            command = request.get("command")
            if command != "compute_embeddings":
                raise ValueError(f"Unsupported command: {command}")

            path_to_bucket = request["path_to_bucket"]
            s3_keys = request["s3_keys"]
            modality_name = request["modality_name"]
            model_name = request.get("model_name")

            if not model_name:
                raise ValueError("Missing 'model_name' in request")

            cache_key = (modality_name, model_name, args.device)
            if cache_key not in model_cache:
                print(f"[worker] Loading model '{model_name}' for modality '{modality_name}' (device={args.device})")
                model = load_model(model_name, modality_name)
                # Device placement, if any, should be handled inside load_model / model itself
                model_cache[cache_key] = model
            else:
                model = model_cache[cache_key]

            if modality_name in ["tomography"]:
                embeddings, checksums, metadata = model.compute_embeddings(
                    s3_connection,
                    s3_keys,
                    path_to_bucket,
                )
            else:
                objects, metadata, checksums, sizes, has_zero = load_data_from_s3(
                    s3_connection,
                    path_to_bucket,
                    s3_keys,
                    modality_name,
                )
                embeddings = model.compute_embeddings(objects)
                if has_zero:
                    zero_out_embeddings_for_empty_objects(embeddings, sizes, modality_name)

            reply = {
                "job_id": job_id,
                "status": "ok",
                "embeddings": embeddings,
                "checksums": checksums,
                "metadata": metadata,
            }
        except Exception as exc:
            print(f"[worker] Error while processing job {job_id}: {exc}")
            reply = {
                "job_id": job_id,
                "status": "error",
                "error": str(exc),
            }

        socket.send_json(reply)


if __name__ == "__main__":
    main()

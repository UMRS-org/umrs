#!/usr/bin/env python3
"""Simple ZeroMQ JSON-RPC client for UMRS ZeroMQ API.

Usage examples:
  python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 health
  python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 status
  python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 upload_text --username user1 --bucket_alias mybucket --text "Hello world"
  python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 upload_image --username user1 --bucket_alias mybucket --file ./data/img_test/cat.jpeg

The client sends a JSON object with keys: {"action": "<action>", "params": {...}} and prints the JSON response.
"""

import argparse
import json
import base64
import os

try:
    import zmq
except Exception:
    zmq = None


def send_request(host, port, payload, timeout=60000):
    if zmq is None:
        raise RuntimeError("pyzmq (zmq) is required to use this client")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    addr = f"tcp://{host}:{port}"
    sock.connect(addr)
    sock.send_json(payload)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    socks = dict(poller.poll(timeout))
    if socks.get(sock) == zmq.POLLIN:
        resp = sock.recv()
        try:
            return json.loads(resp.decode('utf-8'))
        except Exception:
            return {"raw_response": resp}
    else:
        raise TimeoutError(f"No response from server at {addr} within {timeout}ms")


def main():
    parser = argparse.ArgumentParser(description="ZeroMQ JSON-RPC client for UMRS")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8443)
    parser.add_argument('action', help='Action to call on server')
    parser.add_argument('payload_file', nargs='?', help='Optional JSON payload file for admin actions')
    parser.add_argument('--username')
    parser.add_argument('--bucket_alias')
    parser.add_argument('--text')
    parser.add_argument('--file')
    parser.add_argument('--metadata', help='JSON string for metadata', default='{}')
    args = parser.parse_args()

    params = {}
    if args.username:
        params['username'] = args.username
    if args.bucket_alias:
        params['bucket_alias'] = args.bucket_alias
    try:
        params['metadata'] = json.loads(args.metadata or '{}')
    except Exception:
        params['metadata'] = {}

    if args.action == 'upload_text':
        params['text'] = args.text or ''
    elif args.action in ('upload_image', 'upload_textfile'):
        if not args.file:
            raise SystemExit('Error: --file is required for upload_image/upload_textfile')
        if not os.path.exists(args.file):
            raise SystemExit(f'File not found: {args.file}')
        with open(args.file, 'rb') as f:
            b = f.read()
        params['filename'] = os.path.basename(args.file)
        params['file_bytes'] = base64.b64encode(b).decode('ascii')
    elif args.action in ('create_user', 'create_buckets', 'add_user_hot_reload'):
        # Expect a JSON config file path as the positional payload_file
        if not args.payload_file:
            raise SystemExit('Error: payload_file (user config json) is required for admin actions')
        if not os.path.exists(args.payload_file):
            raise SystemExit(f'Payload file not found: {args.payload_file}')
        with open(args.payload_file, 'r', encoding='utf-8') as f:
            try:
                params = json.load(f)
            except Exception as e:
                raise SystemExit(f'Failed to parse payload file: {e}')
    elif args.action in ('recompute', 'update', 'check'):
        # no params expected for these admin actions
        pass

    payload = {'action': args.action, 'params': params}

    try:
        resp = send_request(args.host, args.port, payload)
        print(json.dumps(resp, indent=2))
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == '__main__':
    main()

"""
UMRS API Client: ZeroMQ-only client for UMRS ZeroMQ RPC API.

This client keeps the same CLI commands as the previous HTTP client but
communicates via ZeroMQ (REQ/REP JSON-RPC). It is compatible with
`examples/zeromq_client.py` semantics and the server's action names.
"""

import argparse
import json
import sys
import os
import base64
from typing import Optional

try:
    import zmq
except Exception as e:
    zmq = None


class UMRSApiClientError(Exception):
    pass


class UMRSApiClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 8443, timeout_ms: int = 60000):
        if zmq is None:
            raise UMRSApiClientError('pyzmq (zmq) is required for ZeroMQ client')
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.addr = f'tcp://{self.host}:{self.port}'
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(self.addr)

    def _send(self, action: str, params: Optional[dict] = None) -> dict:
        params = params or {}
        payload = {'action': action, 'params': params}
        try:
            self.sock.send_json(payload)
        except Exception as e:
            raise UMRSApiClientError(f'Failed to send request: {e}')

        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        socks = dict(poller.poll(self.timeout_ms))
        if socks.get(self.sock) == zmq.POLLIN:
            try:
                resp = self.sock.recv()
                try:
                    return json.loads(resp.decode('utf-8'))
                except Exception:
                    return {'raw_response': resp}
            except Exception as e:
                raise UMRSApiClientError(f'Failed to receive response: {e}')
        else:
            raise UMRSApiClientError(f'No response from server at {self.addr} within {self.timeout_ms}ms')

    def add_user_hot_reload(self, user_config: dict):
        return self._send('add_user_hot_reload', user_config)

    def create_user(self, user_config: dict):
        return self._send('create_user', user_config)

    def create_buckets(self, user_config: dict):
        return self._send('create_buckets', user_config)

    def health(self):
        return self._send('health')

    def status(self):
        return self._send('status')

    def list_buckets(self, username: Optional[str] = None):
        params = {'username': username} if username else {}
        return self._send('buckets', params)

    def upload_text(self, text: str, bucket_alias: str, username: str, metadata: Optional[dict] = None):
        params = {'text': text, 'bucket_alias': bucket_alias, 'username': username, 'metadata': metadata or {}}
        return self._send('upload_text', params)

    def upload_image(self, file_path: str, bucket_alias: str, username: str, metadata: Optional[dict] = None):
        if not os.path.exists(file_path):
            raise UMRSApiClientError(f'File not found: {file_path}')
        with open(file_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        params = {'filename': os.path.basename(file_path), 'file_bytes': b64, 'bucket_alias': bucket_alias, 'username': username, 'metadata': metadata or {}}
        return self._send('upload_image', params)

    def upload_textfile(self, file_path: str, bucket_alias: str, username: str, metadata: Optional[dict] = None):
        if not os.path.exists(file_path):
            raise UMRSApiClientError(f'File not found: {file_path}')
        with open(file_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        params = {'filename': os.path.basename(file_path), 'file_bytes': b64, 'bucket_alias': bucket_alias, 'username': username, 'metadata': metadata or {}}
        return self._send('upload_textfile', params)

    def recompute_embeddings(self):
        return self._send('recompute')

    def update_embeddings(self):
        return self._send('update')

    def check_consistency(self):
        return self._send('check')

    def close(self):
        try:
            self.sock.close()
            self.ctx.term()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='UMRS ZeroMQ client CLI')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8443)
    parser.add_argument('--timeout', type=int, default=60000, help='ZeroMQ poll timeout in ms')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('health', help='Check API health')
    subparsers.add_parser('status', help='Check API status')

    p_list = subparsers.add_parser('list_buckets', help='List buckets')
    p_list.add_argument('username', nargs='?')

    p_upload_text = subparsers.add_parser('upload_text', help='Upload text')
    p_upload_text.add_argument('text')
    p_upload_text.add_argument('bucket_alias')
    p_upload_text.add_argument('username')
    p_upload_text.add_argument('metadata', nargs='?')

    p_upload_img = subparsers.add_parser('upload_image', help='Upload image')
    p_upload_img.add_argument('file_path')
    p_upload_img.add_argument('bucket_alias')
    p_upload_img.add_argument('username')
    p_upload_img.add_argument('metadata', nargs='?')

    p_upload_txt = subparsers.add_parser('upload_textfile', help='Upload text file')
    p_upload_txt.add_argument('file_path')
    p_upload_txt.add_argument('bucket_alias')
    p_upload_txt.add_argument('username')
    p_upload_txt.add_argument('metadata', nargs='?')

    subparsers.add_parser('recompute_embeddings', help='Recompute embeddings (sends action "recompute")')
    subparsers.add_parser('update_embeddings', help='Update embeddings (sends action "update")')
    subparsers.add_parser('check_consistency', help='Check consistency (sends action "check")')

    p_add = subparsers.add_parser('add_user_hot_reload', help='Add user hot reload')
    p_add.add_argument('user_config')

    p_create_user = subparsers.add_parser('create_user', help='Create user')
    p_create_user.add_argument('user_config')

    p_create_buckets = subparsers.add_parser('create_buckets', help='Create buckets')
    p_create_buckets.add_argument('user_config')

    args = parser.parse_args()

    client = UMRSApiClient(host=args.host, port=args.port, timeout_ms=args.timeout)

    try:
        if args.command == 'health':
            print(json.dumps(client.health(), indent=2))
        elif args.command == 'status':
            print(json.dumps(client.status(), indent=2))
        elif args.command == 'list_buckets':
            print(json.dumps(client.list_buckets(username=getattr(args, 'username', None)), indent=2))
        elif args.command == 'upload_text':
            metadata = json.loads(args.metadata) if getattr(args, 'metadata', None) else None
            print(json.dumps(client.upload_text(args.text, args.bucket_alias, args.username, metadata=metadata), indent=2))
        elif args.command == 'upload_image':
            metadata = json.loads(args.metadata) if getattr(args, 'metadata', None) else None
            print(json.dumps(client.upload_image(args.file_path, args.bucket_alias, args.username, metadata=metadata), indent=2))
        elif args.command == 'upload_textfile':
            metadata = json.loads(args.metadata) if getattr(args, 'metadata', None) else None
            print(json.dumps(client.upload_textfile(args.file_path, args.bucket_alias, args.username, metadata=metadata), indent=2))
        elif args.command == 'recompute_embeddings':
            print(json.dumps(client.recompute_embeddings(), indent=2))
        elif args.command == 'update_embeddings':
            print(json.dumps(client.update_embeddings(), indent=2))
        elif args.command == 'check_consistency':
            print(json.dumps(client.check_consistency(), indent=2))
        elif args.command == 'add_user_hot_reload':
            with open(args.user_config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            print(json.dumps(client.add_user_hot_reload(cfg), indent=2))
        elif args.command == 'create_user':
            with open(args.user_config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            print(json.dumps(client.create_user(cfg), indent=2))
        elif args.command == 'create_buckets':
            with open(args.user_config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            print(json.dumps(client.create_buckets(cfg), indent=2))
        else:
            print(f'Unknown command: {args.command}', file=sys.stderr)
            sys.exit(1)
    except UMRSApiClientError as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f'[UNEXPECTED ERROR] {e}', file=sys.stderr)
        sys.exit(3)
    finally:
        client.close()


if __name__ == '__main__':
    main()


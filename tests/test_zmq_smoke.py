#!/usr/bin/env python3
import sys
import os
import time
import threading
import tempfile

# Ensure local src/ is importable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, os.pardir, 'src'))

from umrs.MessageQueue import MessageQueue, DEFAULT_QUEUE_NAME


def server_thread(outlist):
    mq = MessageQueue(queue_type='zeromq', role='server', queue_name=DEFAULT_QUEUE_NAME)
    # Server SUB socket is ready immediately after construction; block waiting for file
    recv_dir = os.path.join(BASE_DIR, 'tmp_recv')
    os.makedirs(recv_dir, exist_ok=True)
    received_path = mq.recv_file(dir=recv_dir)
    with open(received_path, 'rb') as f:
        data = f.read()
    outlist.append((received_path, data))
    mq.close()


def client_send(tempfile_path):
    mq = MessageQueue(queue_type='zeromq', role='client', queue_name=DEFAULT_QUEUE_NAME)
    # small pause to let subscription propagate to the publisher
    time.sleep(0.5)
    mq.send_file(tempfile_path, chunk_size=512)
    # client sockets are not closed via MessageQueue.close() (server-only)
    try:
        if hasattr(mq, 'socket'):
            mq.socket.close()
        if hasattr(mq, 'context'):
            mq.context.term()
    except Exception:
        pass


def main():
    # create a temporary test file with deterministic content
    tf = tempfile.NamedTemporaryFile(delete=False)
    try:
        content = (b"Hello ZeroMQ file transfer\n" * 32)
        tf.write(content)
        tf.flush()
        tmp_path = tf.name
    finally:
        tf.close()

    out = []
    t = threading.Thread(target=server_thread, args=(out,))
    t.start()

    client_send(tmp_path)

    t.join(timeout=10)

    if not out:
        print('SMOKE TEST FAIL: no file received')
        sys.exit(2)

    received_path, data = out[0]
    with open(tmp_path, 'rb') as f:
        orig = f.read()

    if orig == data:
        print('SMOKE TEST PASS: received file matches')
        print('received file:', received_path)
        sys.exit(0)
    else:
        print('SMOKE TEST FAIL: content mismatch')
        sys.exit(3)


if __name__ == '__main__':
    main()

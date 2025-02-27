import zmq
from time import sleep

# Global configuration of the message queue
target_nb_messages = 10
target_message_size = 4096  # ZeroMQ handles message size internally

# ZeroMQ context and sockets
context = zmq.Context()

# PUB socket for sending messages
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://127.0.0.1:5555")  # Binding to local port

# SUB socket for receiving messages
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://127.0.0.1:5555")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages


def close_message_queue():
    """Close ZeroMQ sockets and context."""
    publisher.close()
    subscriber.close()
    context.term()


def notify_s3(s3_object):
    """Send a message to the ZeroMQ PUB socket."""
    print(f"🔔 Notifying: {s3_object}")
    publisher.send_string(s3_object)


def wait_for_s3_events(debug=False):
    """Block until new messages arrive, then read all available messages."""
    messages = []

    print("📡 Listening for new messages...")

    # Poller for non-blocking checks
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    # BLOCKING: Wait for the first message
    while True:
        socks = dict(poller.poll())
        if subscriber in socks:
            message = subscriber.recv_string()
            print("✅ Received (blocking):", message)
            messages.append(message)
            break  # Exit after receiving the first message

    # NON-BLOCKING: Check for more messages
    while True:
        socks = dict(poller.poll(timeout=0))
        if subscriber in socks:
            message = subscriber.recv_string()
            messages.append(message)
            if debug:
                print("🔄 Received (non-blocking):", message)
        else:
            if debug:
                print("⚡ No more messages in queue.")
            break

    return messages


# Example Usage
if __name__ == "__main__":
    try:
        sleep(0.5)  # ✅ Give time for subscriber to connect
        # Notify some S3 events
        notify_s3("s3://bucket/file1.txt")
        sleep(0.1)  # Small delay to ensure message delivery
        notify_s3("s3://bucket/file2.txt")
        sleep(0.1)

        # Wait for S3 events
        events = wait_for_s3_events(debug=True)
        print("\n📋 All received events:", events)

    finally:
        # Cleanup message queue
        close_message_queue()

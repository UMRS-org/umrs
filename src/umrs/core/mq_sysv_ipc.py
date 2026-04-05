import sysv_ipc
import warnings
import signal

# Global configuration of the message queue
target_nb_messages = 10
target_message_size = 4096

# Define a unique key for the System V message queue
MQ_KEY = 1234  # You can change this to any unused integer

# Create or connect to the message queue
try:
    mq = sysv_ipc.MessageQueue(MQ_KEY, sysv_ipc.IPC_CREX)  # Create new queue
except sysv_ipc.ExistentialError:
    mq = sysv_ipc.MessageQueue(MQ_KEY)  # Connect if it already exists

print(f"Queue size: {mq.max_size}")

def close_message_queue():
    """Close and remove the message queue."""
    mq.remove()
    print("Message queue removed.")

def notify_s3(s3_object):
    """Send a message to the queue."""
    print(f"Notifying: {s3_object}")
    message = s3_object.encode()  # Convert string to bytes
    if len(message) > target_message_size:
        print(f"Message too large! Max size is {target_message_size} bytes.")
        return
    mq.send(message, block=False)

def wait_for_s3_events(debug=False):
    """Block until a new message arrives, then read all available messages."""
    messages = []

    print("Listening for new messages...")
    try : 
        # Blocking read for the first message
        message, msg_type = mq.receive(type=1)  # Blocking wait for message of type 1
        message_decoded = message.decode()
        print("Received (blocking):", message_decoded)
        messages.append(message_decoded)

        # Non-blocking reads for remaining messages
        while True:
            try:
                message, msg_type = mq.receive(block=False)
                message_decoded = message.decode()
                messages.append(message_decoded)
                if debug:
                    print("Received (non-blocking):", message_decoded)
            except sysv_ipc.BusyError:
                if debug:
                    print("No more messages in queue.")
            break

    except sysv_ipc.Error as e:
        # Handle the 'Signaled while waiting' error silently
        print("\nReceived signal, exiting gracefully...")

    except KeyboardInterrupt:
        # Handling  Ctrl+C interruption is done by the caller.
        pass

        
    return messages

# Example Usage:
if __name__ == "__main__":
    try:
        # Notify some S3 events
        notify_s3("s3://bucket/file1.txt")
        notify_s3("s3://bucket/file2.txt")

        # Wait for S3 events
        events = wait_for_s3_events(debug=True)
        print("All received events:", events)

    finally:
        # Cleanup message queue
        close_message_queue()

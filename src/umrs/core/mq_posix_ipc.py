
import posix_ipc

## Global configuration of the message queue
target_nb_messages = 10
target_message_size = 4096

mq = posix_ipc.MessageQueue(
    "/MQ_Queue",
    posix_ipc.O_CREAT,    # Create the queue if it doesn't exist
    max_messages=target_nb_messages,      # Limit to 100 messages in the queue
    max_message_size=target_message_size  # Each message can be up to 8192 bytes
)

if (mq.max_messages < target_nb_messages or mq.max_message_size < target_message_size):
    print("############# Warning #############")
    print("Configuration of the IPC Message Queue did not match the requested values")
    print(f"\tQueue size (nb of msg) = {mq.max_messages} (requested_value = {target_nb_messages})")
    print(f"\tMessage size = {mq.max_message_size} (requested_value = {target_message_size})")
    print("\tConsider changing the values of /proc/sys/fs/mqueue/msg_max and /proc/sys/fs/mqueue/msgsize_max...\n")



def close_message_queue():
    mq.close()
    mq.unlink()



def notify_s3(s3_object):
    print(f"Notyfying {s3_object}")
    mq.send(s3_object)


def wait_for_s3_events(debug=False):
    """Blocks execution until a new message arrives, then reads all available messages."""

    # BLOCKING: Wait for the first message
    print(f"Listening the queue ({mq.current_messages} pending msg)...")
    message, _ = mq.receive()
    print("Received (blocking):", message.decode())

    messages = [message.decode()]  # Store messages

    if mq.current_messages == 0:
        return messages

    print("More to come...")
    # NON-BLOCKING: Read remaining messages without blocking
    while True:
        try:
            message, _ = mq.receive(timeout=0)  # Non-blocking read
            messages.append(message.decode())
            print("Received (Non blocking):", message.decode())

        except posix_ipc.BusyError:
            if debug:
                print("No more messages in queue.")
            break  # Exit loop if queue is empty

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

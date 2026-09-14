import select

import psycopg2

# Database connection parameters
DB_PARAMS = {
    "dbname": "minio_events",
    "user": "ejeannot",
    "host": "localhost",  # Change if necessary
    "port": "5432",
}


def listen_for_new_entries():
    """Listens for new entries in PostgreSQL using LISTEN/NOTIFY."""
    conn = psycopg2.connect(**DB_PARAMS)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()
    cur.execute("LISTEN new_entry_channel;")  # Subscribe to the channel

    print("👂 Listening for new entries... Press Ctrl+C to exit.")

    try:
        while True:
            # Wait for notifications (blocking)
            if select.select([conn], [], [], None)[0]:
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    print(f"🆕 New entry added with ID: {notify.payload}")

    except KeyboardInterrupt:
        print("\n👋 Listener stopped.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    listen_for_new_entries()

# UMRS

Config-driven server and client for multimodal ingestion into S3 + embeddings in PostgreSQL/pgvector. See the project overview for architecture, Docker usage, and implementation guide:

- doc: `doc/umrs_overview.md`

## Quick start with Docker

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker logs -f umrs-server
```

Then try the client from inside the server container:

```bash
docker exec -it umrs-server python src/client_umrs.py ping
docker exec -it umrs-server python src/client_umrs.py --conf /root/conf_umrs.json list_buckets user1
```

More examples and details are in `doc/umrs_overview.md` and `docker/README.md`.

## Additional docs

- `doc/HOWTO.md` – legacy how-to notes
- `doc/set-up_pgvector.md` – details for pgvector
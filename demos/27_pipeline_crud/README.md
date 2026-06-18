# Demo 27: Pipeline CRUD & RAG Profiles

## Purpose

Demonstrates the UMRS pipeline management API — creating, listing,
updating, and deleting pipelines, plus RAG profile composition
(chunker + embedder + reranker) and the component registry.

## API Endpoints Covered

| Method | Path                                      | Description              |
|--------|-------------------------------------------|--------------------------|
| GET    | `/umrs/pipelines`                         | List pipelines           |
| POST   | `/umrs/pipelines`                         | Create pipeline          |
| GET    | `/umrs/pipelines/{id}`                    | Get pipeline             |
| PUT    | `/umrs/pipelines/{id}`                    | Update pipeline          |
| DELETE | `/umrs/pipelines/{id}`                    | Delete pipeline          |
| GET    | `/umrs/pipelines/rag-profiles`            | List RAG profiles        |
| POST   | `/umrs/pipelines/rag-profiles`            | Create RAG profile       |
| GET    | `/umrs/pipelines/components/chunkers`     | List chunkers            |
| GET    | `/umrs/pipelines/components/embedders`    | List embedders           |
| GET    | `/umrs/pipelines/components/rerankers`    | List rerankers           |

## Run

```bash
# Follow the step-by-step commands in README.md
```

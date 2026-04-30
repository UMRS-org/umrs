# Demo 25: Endpoint Provisioning API

## Purpose

Demonstrates the UMRS endpoint provisioning REST API — managing
Mini-RiVault endpoint lifecycle including warm pool reuse,
multi-model co-location, and LoRA adapter support.

## Prerequisites

- UMRS server running (HTTP API on port 8001)
- At least one model registered or pre-staged

## API Endpoints Covered

| Method | Path                       | Description                |
|--------|----------------------------|----------------------------|
| POST   | `/umrs/endpoints/provision`| Provision a new endpoint   |
| POST   | `/umrs/endpoints/up`       | Alias for provision        |
| GET    | `/umrs/endpoints`          | List active endpoints      |
| DELETE | `/umrs/endpoints/{id}`     | Stop an endpoint           |

## Run

```bash
bash run_demo.sh
```

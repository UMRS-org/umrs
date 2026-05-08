# Demo 30: File Management API

## Purpose

Demonstrates the UMRS file management REST API — upload, bulk upload,
download (with presigned URLs), and deletion of files stored in MinIO/S3.

## API Endpoints Covered

| Method | Path                          | Description                 |
|--------|-------------------------------|-----------------------------|
| POST   | `/umrs/files/upload`          | Upload single file          |
| POST   | `/umrs/files/bulk-upload`     | Upload multiple files       |
| GET    | `/umrs/files/{file_key}`      | Download / get presigned URL|
| DELETE | `/umrs/files/{file_key}`      | Delete a file               |

## Run

```bash
bash run_demo.sh
```

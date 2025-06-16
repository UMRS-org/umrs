# U.M.R.S. Demo Guide

## 1. Prepare the Environment

### ➤ Switch to the demo branch

`git checkout demo`

### ➤ [Optional] Launch the local S3 service (e.g., MinIO)

`minio server /data --console-address ":9001"`

## 2. Terminal Windows Setup

### Window 1 — PostgreSQL

1. Connect to PostgreSQL and drop existing demo databases:

```
psql -U postgres
DROP DATABASE IF EXISTS images_and_texts_ejeannot;
DROP DATABASE IF EXISTS text_and_images_ejeannot;
```

### Window 2 — Server

1. Launch the U.M.R.S. server:

`./server_umrs.py`

Wait for the server to finish setup and listen for incoming requests.

### Window 3 — Client

## 3. Upload and Query Data

### 3.1 Submit text examples
```
./client_umrs.py "I eat apple at breakfast" ej-bucket-2
./client_umrs.py "I like apple and any fruit!" ej-bucket-2 -k keywords:test
```

### 3.2 Query SQL to verify

```
psql -d text_and_images_ejeannot
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```

### 3.3 Upload a full text file

`./client_umrs.py apple.txt ej-bucket-2`

### 3.4 Check results in DB

`SELECT s3_object, _content, date, time FROM text ORDER BY time;`

### 3.5 Cluster texts using PGVector

`../test_embeddings/cluster_txt.py`

## 4. Work with Images

### 4.1 Upload an entire image directory

`./client_umrs.py ../../data/img ej-bucket-1`

### 4.2 Query the image table

`SELECT s3_object, date, time FROM image ORDER BY time;`

### 4.3 Cluster images using PGVector

`../test_embeddings/cluster_img.py`

## 5. Metadata & Keywords

### Upload images with/without keywords

```
./client_umrs.py ../../data/img/Cat_2.jpeg ej-bucket-3
./client_umrs.py ../../data/img/Cat_1.jpeg ej-bucket-3 -k keywords:cat -k author:internet -k x_resolution:256 -k y_resolution:512
```

### Query metadata

```
psql -d images_and_texts_ejeannot
SELECT id, s3_object, date, time, keywords, author, x_resolution, y_resolution FROM image ORDER BY time;
```

## 6. Upload a Full Text File as a Document

### Submit a large file

`./client_umrs.py ../../data/texts/beast_in_the_jungle.txt ej-bucket-5`
### Inspect chunking results

```
psql -d images_and_texts_ejeannot
SELECT s3_object, _content, date, time, chunk_start, chunk_end, chunk_md5 FROM text_file;
```

## 7. Wrap Up

### Return to the main branch

`git checkout main`

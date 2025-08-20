# U.M.R.S. Demo Guide

## 1. Prepare the Environment

### ➤ Switch to the demo branch

`git checkout demo`

### ➤ [Optional] Launch the local S3 service (e.g., MinIO)

`minio server /data --console-address ":9001"`

### ➤ Delete minIO users
`mc admin user rm myminio user1 --insecure`
`mc admin user rm myminio user2 --insecure`


## 2. Terminal Windows Setup

### Window 1 — PostgreSQL

1. Connect to PostgreSQL and drop existing demo databases and delete users:

```
psql -U postgres
\t 
drop database images_and_texts_usr1_1;
drop database images_and_texts_usr2_1;
drop database text_and_images_usr1_2;
\du 
drop user usr1_1;
drop user usr1_2;
drop user usr2_1;
```

### Window 2 — Server

1. Launch the U.M.R.S. server:

`./server_umrs.py`

Wait for the server to finish setup and listen for incoming requests.

### Window 3 — Client

## 3. Upload and Query Data

### 3.1 Submit text examples
```
./client_umrs.py -c users_json/user1.json "I eat an apple at breakfast" bucket-2
./client_umrs.py -c users_json/user2.json "I eat an apple at breakfast" bucket-2

```

### 3.2 Query SQL to verify

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```

check that credantial works and that you cannot read the database if you are not the right user: 

```
psql -d text_and_images_usr1_2 usr1_1
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```
### 3.3 Upload a full text file

`./client_umrs.py -c users_json/user1.json apple.txt bucket-2`

### 3.4 Check results in DB

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, _content, date, time FROM text ORDER BY time;
```

### 3.5 Cluster texts using PGVector

`../test_embeddings/cluster_txt.py`

## 4. Work with Images

### 4.1 Upload an entire image directory

` ./client_umrs.py -c users_json/user1.json ../../data/img bucket-1`

### 4.2 Query the image table

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, date, time FROM image ORDER BY time;
```

### 4.3 Cluster images using PGVector

`../test_embeddings/cluster_img.py`

## 5. Metadata & Keywords

### Upload images with/without keywords

```
./client_umrs.py -c users_json/user2.json ../../data/img/Cat_2.jpeg  bucket-1
./client_umrs.py ../../data/img/Cat_1.jpeg ej-bucket-3 -k keywords:cat -k author:internet -k x_resolution:256 -k y_resolution:512
```

### Query metadata

```
psql -d images_and_texts_usr2_1  usr2_1
SELECT id, s3_object, date, time, keywords, author, x_resolution, y_resolution FROM image ORDER BY time;
```

## 6. Upload a Full Text File as a Document

### Submit a large file

`./client_umrs.py -c users_json/user1.json "I eat an apple at breakfast" bucket-2`
### Inspect chunking results

```
psql -d images_and_texts_usr1_1 usr1_1
```

```
SELECT _content, chunk_start, chunk_end, chunk_md5
FROM text_file
WHERE s3_object ~* 'beast'
ORDER BY chunk_start;
```


## 7. Wrap Up

### Return to the main branch

`git checkout main`

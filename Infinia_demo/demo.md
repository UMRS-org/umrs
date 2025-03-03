# use the demo branch
git checkout demo

# Create three windows
## Window 1: SQL
1/ connect to the sql server to drop the database
psql -U postgres
drop database  images_and_texts_ejeannot;
drop database  text_and_images_ejeannot;

## server window
1/ ./server_umrs.py
2/ wait for the server to configure and blocks in messages from the client


## client window
1/ add a text

./client_umrs.py "I eat apple at breakfast" ej-bucket-2 
./client_umrs.py "I like apple and any fruit!" ej-bucket-2 -k keywords:test

2/ show the results on the server side and the SQL window:
psql -d text_and_images_ejeannot
select s3_object,_content,keywords,date,time from text order by time;

3/ add all the lines of file:
./client_umrs.py  apple.txt ej-bucket-2

4/ sho the results on the server side and the SQL window:
select s3_object,text,date,time  from text order by time;

5/ show clustering capabilities with PG Vectors
../test_embeddings/cluster_txt.py


6/ add a full directory
  ./client_umrs.py  ../../data/img ej-bucket-1

7/ check the DB
select s3_object,date,time  from image order by time;

8/ show clustering capabilities with PG Vectors
../test_embeddings/cluster_img.py

9/show keyword management: 
./client_umrs.py ../../data/img/Cat_2.jpeg ej-bucket-3 
./client_umrs.py ../../data/img/Cat_1.jpeg ej-bucket-3 -k keywords:cat -k author:internet -k x_resolution:256 -k y_resolution:512

psql -d images_and_texts_ejeannot
select id,s3_object,date,time,keywords,author,x_resolution,y_resolution  from image order by time;

# Return to the main branch
git checkout main

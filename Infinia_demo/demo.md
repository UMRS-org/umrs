# Use the demo branch
git checkout demo_test

# Create three windows
## Window 1: SQL
1/ connect to the sql server to earse the XP database
psql -d text_and_images_ejeannot
TRUNCATE TABLE text CASCADE;
TRUNCATE TABLE image CASCADE;

2/ check it is empty
select * from image;
select * from text;

## server window
1/ ./server_umrs.py conf_umrs.json
2/ wait for the server to configure and blocks in messages from the client


## client window
1/ add a text

 ./client_umrs.py  -u ejeannot -x "Text and Images" -m text -k "Apple" -t text -d "I like eating apples during breakfast"

2/ show the results on the server side and the SQL window:
select s3_object,text,date,time  from text order by time;

3/ add all the lines of file:
./client_umrs.py  -u ejeannot -x "Text and Images" -m text -k "Apple" -t text -d apple.txt

4/ sho the results on the server side and the SQL window:
select s3_object,text,date,time  from text order by time;

5/ show clustering capabilities with PG Vectors
../test_embeddings/cluster_txt.py

6/ add an imag
  ./client_umrs.py  -u ejeannot -x "Text and Images" -m image -k "Apple" -t img/jpg -d ../../data/img/Cat_1.jpeg

7/ check the DB
select s3_object,date,time  from image order by time;

8/ add a full directory
  ./client_umrs.py  -u ejeannot -x "Text and Images" -m image -k "Apple" -t img/jpg -d ../../data/img

9/ show clustering capabilities with PG Vectors
../test_embeddings/cluster_img.py


# Return to the main branch
git checkout main

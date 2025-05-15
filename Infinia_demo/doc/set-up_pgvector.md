# Setting Up PostgreSQL & PGVector on Ubuntu

## 1 Install PostgreSQL
```sh
sudo apt install -y postgresql postgresql-contrib
```

## 2️Create a PostgreSQL User (`ejeannot`) with Superuser Privileges
```sh
sudo -i -u postgres
psql
```
```sql
CREATE ROLE ejeannot WITH LOGIN;
ALTER ROLE ejeannot CREATEDB;
ALTER ROLE ejeannot SUPERUSER;
```
```sh
\q
```

## 3️Enable Passwordless Authentication (Optional)
```sh
sudo vi /etc/postgresql/*/main/pg_hba.conf
```
```
local   all   all   peer  →  local   all   all   trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
```


## 4  Restart PostgreSQL to Apply Changes
```sh
sudo systemctl restart postgresql
```
You then should be able to connect to the psql without a password:
```sh
psql -U ejeannot -d postgres
```


## 5️ Install PGVector for PostgreSQL
```sh
sudo apt install -y postgresql-server-dev-all
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make -j 32 
sudo make install
```

## 6  Enable PGVector in PostgreSQL
```sh
psql -U ejeannot -d postgres -c "CREATE EXTENSION vector;"
```

**PGVector is now installed and ready to use!**

## 7 MISC
```sh
sudo apt install -y python3-boto3
sudo apt install -y python3-psycopg2
sudo apt install -y python3-pil
sudo apt install -y python3-numpy
sudo apt install python3-pip
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install sentence-transformers --break-system-packages
sudo apt install python3-zmq
sudo apt install python3-pandas
sudo apt install python3-tabulate
```

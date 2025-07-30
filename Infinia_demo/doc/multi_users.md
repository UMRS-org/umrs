# UMRS: Unified Multimodal Retriever Service

## Multiuser Documentation

### 1. Configuration Files

- **Per-User Configuration**  
  Each user has their own configuration file, for example in `src/users_json/`:
    - `user1.json`
    - `user2.json`

- **Master Configuration**  

There is a master configuration file that lists all individual user files:

```
	{
      "admin_s3_configuration_file": "~/.s3cfg",
      "users": [
        "json/user1.json",
        "json/user2.json"
      ]
    }
```
- **File Structure**  
  Each user JSON user file describes the user’s experiments and modalities.

- **Resources**  
    - One S3 bucket per modality.
    - One database per experiment.
    - One table per modality in each database.

- **User Credentials**  
  There is single a single S3 user but there may be different SQL users (one per database).  
  The user JSON indicates usernames, modalities, experiment names, and the path to the user’s S3 config file

- The master JSON also holds the S3 admin configuration file path, which is used for server identification and admin connection.

### 2. Server Initialization

When started, the server will:

- Read every user’s configuration JSON file.
- Create any needed SQL users (for now SQL user password if the same as the SQL username).
- Create the S3 user (if not present).
- Create all required buckets, then set access policies so:
    - Each user can access only their own bucket.
    - Users cannot access others’ buckets.

### 3. Server Configuration and Mappings

- The server configures all database and S3 connections.
- It maps each bucket to the correct table according to the user configuration files.
- Bucket aliases can be reused by different users, since they are isolated in different config files.

### 4. Client Workflow

- Clients must use their own user config file (e.g. `user1.json`), not the master config.
- The master config is private and must NOT be distributed—it contains sensitive admin S3 credentials.
- The client configuration links bucket aliases to real buckets and notifies the server after writing new data.
- The server then processes the data, computes embeddings, and stores them in the correct table (by modality/experiment).

### 5. Naming Rules

- **Bucket Alias**  
  Chosen by the user, must be unique per user.

- **Bucket Path**  
  `<s3_username>.<modality>.<bucket_alias>`  
  (All spaces and underscores are replaced by hyphens.)

- **Database Name**  
  `<modality>_<sql username>`  
  (Spaces replaced by underscores.)

### 6. S3 Server Configuration (MinIO)

- S3 user creation and policy setup **must use command-line tools** like `mc`, not boto3.
- These commands are server-specific.  
  **Currently, only MinIO is supported via mc.**

- **TODO:** Port these parts to Infinia.

### 7. Credential Management

- **S3:** Each user can access only their own bucket.
- **SQL:** Each user can access only their own database (their experiments).


# UMRS: Unified Multimodal Retriever Service

## Multiuser Documentation

### 1. Configuration Files

- **Per-User Configuration**  
  Each user has their own configuration file, for example in `src/users_json/`:
    - `user1.json`
    - `user2.json`

Each user must run their own client using their **personal configuration file** (such as `user1.json` for user1). 

When launching the client program, the user must provide their specific configuration file. For example:


- **Server Configuration**  

There is a master configuration file that lists all individual user files:

```
	{
      "admin_s3_configuration_file": "~/.s3cfg",
  "user_files_dir": "json",
      "users": [
        "json/user1.json",
        "json/user2.json"
      ]
    }
```

✅ **Do NOT use the admin configuration file** for client operations.  
The admin config is used exclusively by the server and contains sensitive admin credentials.

🔒 This setup ensures:
- Each user is isolated to their permitted resources (S3 buckets, SQL databases, etc.)
- The client has access only to the correct experiment and modality definitions
- The system behaves securely in a multiuser environment

- **File Structure**  
  Each user JSON user file describes the user’s experiments and modalities.

- **Resources**  
    - One S3 bucket per modality.
    - One database per experiment.
    - One table per modality in each database.


- **S3 Configuration Files**  
  You only need to set up the S3 configuration file for the *admin* user, as specified by `admin_s3_configuration_file`.  
  The individual S3 configuration files for each user do not need to be created manually — they will be automatically generated at the first launch of the server.
  Once the s3 configuration files of each users are created they should be put in the path `s3_configuration_file`on the client machine.  
   
- **File Creation Workflow**
	1. **Create the admin S3 configuration file**  
   This file (e.g. `~/.s3cfg`) will be used by the server to manage S3 service acces (e.g. miniIO or Infinia) access and create user resources.
   2. **Create the per-user JSON configuration files**  
   Each user should have their own config file (e.g. `user1.json`, `user2.json`) describing experiments, modalities, etc.
   3. **Create the main server configuration file**  
   This file (e.g. `conf_umrs.json`) should list all the user configuration files and specify the admin S3 config path (as explained above) 
   4. **Launch the server**  
   The server will automatically generate S3 user configuration files for each user listed in the server configuration files
   5. **Distribute user configuration files to client machines**  
   For each user:
   		- Copy the user's JSON config file and the auto-generated S3 config file to the corresponding client machine.
   		- Optionally rename the user's config to `conf_umrs.json` on the client.  
    	- If not renamed, the client must be launched using the `-c` or `--conf` option to specify the file path.
    6. **Launch the client on each user's machine**  
   This will allow the user to push data into their assigned S3 bucket (by alias), triggering embedding and storage on the server.

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


### 7. Adding a New User

To add a new user with their own set of experiments and modalities:

1. **Create the user's configuration file**  
   Define the user's experiments, modalities, database settings, and S3 bucket aliases in a new config file (e.g. `user3.json`).

2. **Update the server configuration file**  
   Add the new user config path to the `users` array in the admin server config (`conf_umrs.json`):
   ```
   {
     "admin_s3_configuration_file": "~/.s3cfg",
     "user_files_dir": "users_json",
     "users": [
       "users_json/user1.json",
       "users_json/user2.json",
       "users_json/user3.json"
     ]
   }
   ```

3. **Restart or re-run the server**  
   The server will detect the new user definition and automatically generate the corresponding S3 user configuration files.

4. **Deploy to the client machine**  
   Copy both of the following to the new user's machine:
   - The user JSON file (e.g. `user3.json`)
   - The generated S3 config file (e.g. `.s3cfg_user3`)

   The user can rename the JSON config as `conf_umrs.json`, or launch the client with:
   ```
   python client_umrs.py --conf user3.json <data> <bucket alias>
   ```

✅ The new user is now ready to push data to their own buckets using the configured client.


### 8. Credential Management

- **S3:** Each user can access only their own bucket.
- **SQL:** Each user can access only their own database (their experiments).


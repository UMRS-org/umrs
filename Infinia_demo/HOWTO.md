# HOWTO

1. **Install a local S3 service**

   I use [MinIO](https://min.io/). Configure it via `~/.s3cfg` like so:

   ```ini
   [default]
   host_base = 127.0.0.1:9000
   host_bucket = 127.0.0.1:9000
   bucket_location = ap-northeast-1
   use_https = False
   check_ssl_certificate = False
   access_key = <your_access_key>
   secret_key = <your_secret_key>
   ```

   Make sure the connection works (e.g. with `s3cmd ls`).

2. **Install PostgreSQL + pgVector**

   - Create a PostgreSQL user. If named `ejeannot`, it will match the existing config; otherwise, you’ll need to adjust it manually.
   - Disable password prompts (no authentication currently used in the scripts).
   - Test that you can create databases and insert data without a password.

3. **Project structure and core scripts**

   - All development happens in the `Infinia_demo` directory.
   - In the `src/` folder, you'll find two main scripts:
     - `client.py`: uploads data to the S3 service.
     - `server.py`: listens for events and triggers embedding computation.

   Both scripts use the config file `conf_umrs.json`.

4. **Use the `demo` branch**

   It provides a reproducible setup and usage guide. see `demo.md` file. If it runs successfully, your environment is properly configured.

5. **Modular components**

   The `src/umrs/` subdirectory contains the Python modules for:
   - S3 interaction
   - PostgreSQL access
   - Task queues
   - Embedding logic, etc.

6. **Model downloads**

   Models should be automatically downloaded via the scripts (e.g. from HuggingFace). Manual download might still be necessary in some setups.

7. **Installation help**

   For tools like MinIO, PostgreSQL, and pgVector, I use ChatGPT for installation steps—feel free to do the same.

8. **Live support**

   I'm available to help you live if needed—just let me know.

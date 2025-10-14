# Deploying configuration file
`./deploy_cfg.sh` to deploy example configuration file to `~/.config/umrs`: 

The directory is as follow: 

``` 
umrs/
├── conf_umrs_full.json
├── conf_umrs_spring8.json
├── conf_umrs.json
├── tmp
└── users
    ├── spring8.json
    ├── user1.json
    └── user2.json
```
 



The root folder `umrs/` mainly contains JSON configuration files and two directories:

- **UMRS Configuration Files**
  - `conf_umrs.json`: default general configuration, used for deployments of user1 and user2.
  - `conf_umrs_spring8.json`: configuration specific to the Spring-8 context, used only for spring8 user model.
  - `conf_umrs_simple.json`: simpler configuration used only for user1.

- **`tmp/` directory**
  - Used to store temporary files, logs, or intermediate data during the service execution.

- **`users/` directory**
  - Contains user profiles or personalized configurations.
  - `spring8.json`: particular configuration for the Spring-8 user (with MAE model).
  - `user1.json` and `user2.json`: profiles and specific configurations for different users.
    


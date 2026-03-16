# UMRS Configuration Examples

## Deploying configuration file
`./deploy_cfg.sh` to deploy example configuration files to `~/.config/umrs`: 

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
  - `spring8.json`: configuration for the Spring-8 user with two tomography pipelines: the legacy MAE embedding model (`tomography_embedding`) and the HMAE (Hierarchical MAE) reconstruction and feature extraction models (`tomography_reconstruction`, `tomography_mae_reconstruction`, `tomography_features_extraction`).
  - `user1.json` and `user2.json`: profiles and specific configurations for different users.

## See Also

- [UMRS Overview](../../doc/umrs_overview.md) — Full description of the configuration model
- [Multi-User Guide](../../doc/multi_users.md) — How to configure multiple users


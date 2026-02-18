# Pull or Update
```
git clone --single-branch --branch umrs git@github.com:RIKEN-RCCS/internal_genAI.git umrs_rivault/
```

# Run on R-CCS Cloud
```
cd umrs_rivault/
sbatch --time=01:00:00 -N 1 -p a100 serve-model.slurm

### after a while we can test if litellm is accessible from login node:
# MODEL_ENDPOINT=$(scontrol show job \
#      $(sacct -u $USER --format=JobID -n -S $(date -d "yesterday" +%Y-%m-%d) | tail -n 1 | awk -F'.' '{print $1}') \
#      | /bin/grep ' NodeList' | awk -F'=' '{print $2}')
#
# curl --connect-timeout 3600 \
#      -H "Authorization: Bearer $(/bin/grep '^LITELLM_MASTER_KEY=' ./env.* |/bin/cut -d'=' -f2)" \
#      "http://${MODEL_ENDPOINT}:$(/bin/grep '^NGINX_LITELLM_DOCKER_PORT=' ./env.* |/bin/cut -d'=' -f2)/v1/models"
#
# rm env.* coredns_fs/coredns.conf.* nginx_fs/out-nginx.conf.*
```

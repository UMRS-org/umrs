#!/bin/bash

mkdir -p "${HOME}/.config/containers/"

cat <<EOF > "${HOME}/.config/containers/storage.conf"
[storage]
driver="overlay"
runroot = "${XDG_RUNTIME_DIR}/containers/storage"
rootless_storage_path="${XDG_RUNTIME_DIR}/containers/storage"
graphroot = "${XDG_RUNTIME_DIR}/containers/storage"     # Note: this shoud be in "${HOME}/.local/share/containers/storage" but NFS prevents it -> now pull every time
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="xattr_permissions=2,threaded=1"
force_mask="700"
EOF

uuid=$(cat /proc/sys/kernel/random/uuid)
cat <<EOF > "./env.${uuid}"
###MANDATORY
VLLM_MODEL=Qwen/Qwen3-0.6B
VLLM_PATH_TO_HUGGINGFACE=./models/vllm/huggingface
VLLM_TP=1						# tensor parallelism
VLLM_PP=1						# pipeline parallelism
NGINX_LITELLM_DOCKER_PORT=$(shuf -i 1025-65535 -n 1)
LITELLM_SALT_KEY=$(openssl rand -hex 8)
LITELLM_MASTER_KEY=$(openssl rand -hex 8)
UMRS_S3_BUCKET_HOST=${SLURM_SUBMIT_HOST}		# httpforever.com
UMRS_S3_BUCKET_PORT=9000
PRIV_IP1TO3=192.168.4
CONTAINER_RUNTIME_DNS=/etc/resolv.conf			# Note: docker uses 127.0.0.11 while podman forces me to use /etc/resolv.conf
UUID=00000000-0000-0000-0000-000000000000
###OPTIONAL
#VLLM_GPU_MEM_UTIL=
#VLLM_MODEL_TIKTOKEN=
#VLLM_CPU_KVCACHE_SPACE=
#VLLM_REPLICAS=
#VLLM_CPU_LIMIT=
#VLLM_MEM_LIMIT=
#MCPO_DOCKER_TAG=
#MCPO_CPU_LIMIT=
#MCPO_MEM_LIMIT=
#NGINX_DOCKER_TAG=
#NGINX_OUT_IP4=
#NGINX_REPLICAS=
#NGINX_CPU_LIMIT=
#NGINX_MEM_LIMIT=
#COREDNS_DOCKER_TAG=
#COREDNS_IP4=
#COREDNS_REPLICAS=
#COREDNS_CPU_LIMIT=
#COREDNS_MEM_LIMIT=
#LITELLM_DOCKER_TAG=
#LITELLM_POSTGRES_USER=
#LITELLM_POSTGRES_PASSWORD=
#LITELLM_POSTGRES_DB=
#LITELLM_MAX_REQUESTS_BEFORE_RESTART=
#LITELLM_REPLICAS=
#LITELLM_CPU_LIMIT=
#LITELLM_MEM_LIMIT=
#POSTGRES_VERSION=
#POSTGRES_REPLICAS=
#POSTGRES_CPU_LIMIT=
#POSTGRES_MEM_LIMIT=
#GENAI_TZ=
VLLM_NCCL_MEM_LIMIT=8G
VLLM_RCCL_MEM_LIMIT=8G
EOF

GPUoverride=()
if /bin/rocm-smi -i | /bin/grep -qi '^GPU\[0\]' ; then
	GPUoverride+=("--file" "./x_rocm-override.yaml")
elif /bin/nvidia-smi --list-gpus | /bin/grep -qi '^GPU 0:' ; then
	GPUoverride+=("--file" "./x_nvidia-override.yaml")
fi

which podman-compose >/dev/null 2>&1 || pip3 install podman-compose

## download model example
#mkdir -p ./models/vllm/huggingface/modules
#pip3 install huggingface_hub
#python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"Qwen/Qwen3-0.6B\", ignore_patterns=[\"metal/*\", \"original/*\"], cache_dir=\"./models/vllm/huggingface/hub\")"
#ls -la ./models/vllm/huggingface/modules/../hub/

# fix some cgroup stuff
loginctl enable-linger $(whoami)

# start
for IP1TO3 in '192.168.4' '172.16.4' '10.0.4'; do
	# Note: --disable-dns needed because ops retards, but now we need fixed ips
	podman network create --internal --disable-dns --subnet "${IP1TO3}.0/24" genai-private-net && break
done
sed -i  -e "s@^\(PRIV_IP1TO3=\).*@\1${IP1TO3}@g" \
	-e "s@^\(UUID=\).*@\1${uuid}@g" \
	"./env.${uuid}"

sed	-e "s@UMRS_S3_BUCKET_PORT@$(/bin/grep '^UMRS_S3_BUCKET_PORT=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./nginx_fs/out-nginx.conf" > "./nginx_fs/out-nginx.conf.${uuid}"
sed     -e "s@UMRS_S3_BUCKET_HOST@$(/bin/grep '^UMRS_S3_BUCKET_HOST=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	-e "s@CONTAINER_RUNTIME_DNS@$(/bin/grep '^CONTAINER_RUNTIME_DNS=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./coredns_fs/coredns.conf" > "./coredns_fs/coredns.conf.${uuid}"

podman compose --env-file "./env.${uuid}" \
               --file "./serve-model.yaml" ${GPUoverride[@]} \
               up \
	       --remove-orphans --detach

podman compose --env-file "./env.${uuid}" \
               --file "./userdata-access.yaml" \
               up \
	       --remove-orphans --detach

ip a
sleep infinity

# test
#curl -H "Authorization: Bearer $(/bin/grep '^LITELLM_MASTER_KEY=' "./env.${uuid}" |/bin/cut -d'=' -f2)" \
#        "http://localhost:$(/bin/grep '^NGINX_LITELLM_DOCKER_PORT=' "./env.${uuid}" |/bin/cut -d'=' -f2)/v1/models"
#podman exec -it $(podman ps -a|/bin/grep vllm|cut -d' ' -f1) curl -v httpforever.com:9000	#XXX: need 'http://$host:80' in out-nginx.conf

# stop
#podman compose --env-file "./env.${uuid}" \
#               --file "./userdata-access.yaml" down
#podman compose --env-file "./env.${uuid}" \
#               --file "./serve-model.yaml" ${GPUoverride[@]} down
#
#podman network rm -f genai-private-net
#
#rm -f "./env.${uuid}" "./nginx_fs/out-nginx.conf.${uuid}" "./coredns_fs/coredns.conf.${uuid}"

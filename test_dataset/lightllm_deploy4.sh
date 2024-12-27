#!/bin/bash

gpus=$1
# 工作空间
workspace_name=$2
# 分区
pool=$3
# 镜像
container_image_url=$4



# INFER=$(pwd)/infer.sh
INFER=$(pwd)/infer1.sh


# 根据 gpus 的值设置 request_mem
if [[ $gpus -eq 1 ]]; then
    request_cpu=12
    request_mem=120
elif [[ $gpus -eq 2 ]]; then
    request_cpu=24
    request_mem=240
elif [[ $gpus -eq 4 ]]; then
    request_cpu=48
    request_mem=480
elif [[ $gpus -eq 8 ]]; then
    request_cpu=96
    request_mem=960
else
    # 默认值或错误处理
    echo "Unsupported gpus value: $gpus"
    exit 1
fi


# 获取当前日期
current_date=$(date +"%Y%m%d")
# # 从模型路径中提取模型名称
# model_name=$(basename $MODEL_PATH)

# 动态生成 app-display-name
# app_display_name="lightllm-${current_date}-${dataset_id}-${model_name}"
app_display_name="lightllm-${current_date}-${dataset_id}"
echo "$app_display_name"

tamp=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6 ; echo '')



command="sco acp jobs create --workspace-name=$workspace_name \
    -p $pool  \
    --container-image-url $container_image_url \
     \
    --storage-mount ef9e6157-1f8e-11ee-88d0-c6880f6d70d9:/mnt/afs,9ec12cab-664b-11ee-ad94-e6e9d682052e:/mnt/pubdata  \
    --worker-spec N3lS.Ii.I60.$gpus  -f pytorch -j $app_display_name --command=\"$INFER \""
echo "model_name: $model_name"


echo "$command"

eval $command | tee $log_file
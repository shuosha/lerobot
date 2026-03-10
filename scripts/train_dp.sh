repo_id=$1
job_name=$2 

# python src/lerobot/scripts/lerobot_train.py \
#     --config_path src/lerobot/rrl/dexgen_config.json \
#     --dataset.repo_id $repo_id \
#     --output_dir outputs/train/${repo_id}_${job_name} \
#     --job_name ${repo_id}_${job_name} \

python src/lerobot/scripts/lerobot_train.py \
    --config_path src/lerobot/rrl/diffusion_config.json \
    --dataset.repo_id $repo_id \
    --output_dir outputs/train/${repo_id}_${job_name} \
    --job_name ${repo_id}_${job_name} \
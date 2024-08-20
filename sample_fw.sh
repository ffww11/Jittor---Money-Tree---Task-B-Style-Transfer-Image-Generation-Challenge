source /root/miniconda3/etc/profile.d/conda.sh

# group=8
weight="best_model816v4+5"
seed_random=0
timestamp=$(date +%s)

echo "Activating environment jittt2..."
conda activate jittt2
echo "Running sample_json.py..."
CUDA_VISIBLE_DEVICES=0 python sample_json_fw.py --group=0 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_fw.py --group=1 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_fw.py --group=8 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_fw.py --group=16 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_fw.py --group=17 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_fw.py --group=18 --weight=$weight --seed_random=$seed_random --time=$timestamp;

echo "sample_json.py finished."
# echo "Activating environment jittt2..."
# conda activate jittt2
# echo "Running sample_json.py..."
# CUDA_VISIBLE_DEVICES=2 python sample_json_fw.py --group=$group --weight=$weight --seed_random=$seed_random --time=$timestamp
# echo "sample_json.py finished."
# cd ../score_fw
# echo "Activating environment jdiffusion_score..."
# conda activate jdiffusion_score
# echo "Running Sum_single.py..."
# CUDA_VISIBLE_DEVICES=3 python Sum_single.py --group=$group --weight=$weight --time=$timestamp
# echo "Sum_single.py finished."

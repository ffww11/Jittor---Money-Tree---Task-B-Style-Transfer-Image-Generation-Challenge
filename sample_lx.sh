source /root/miniconda3/etc/profile.d/conda.sh

weight="best_model816v4+5"
seed_random=0
timestamp=$(date +%s)
echo "Activating environment jittt2..."
conda activate jittt2
echo "Running sample_json.py..."
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=2 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=3 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=4 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=10 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=11 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=12 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=19 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=20 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=21 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=22 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=23 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=24 --weight=$weight --seed_random=$seed_random --time=$timestamp;
CUDA_VISIBLE_DEVICES=0 python sample_json_lx.py --group=26 --weight=$weight --seed_random=$seed_random --time=$timestamp;
echo "sample_json.py finished."

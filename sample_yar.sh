source /root/miniconda3/etc/profile.d/conda.sh

groups=(5 6 7 9 13 14 15 25 27)  # 这里可以添加多个group编号
# groups=(13)
weight="best_model816v4+5"
seed_random=0
timestamp=$(date +%s)

for group in "${groups[@]}"; do
  echo "Processing group $group..."

  conda activate jittt2
  CUDA_VISIBLE_DEVICES=0 python sample_yar.py --group=$group --weight=$weight --seed_random=$seed_random --time=$timestamp
  echo "sample finished for group $group."

#  conda activate jdiffusion_score
#  CUDA_VISIBLE_DEVICES=1 python Sum_single_yar.py --group=$group --weight=$weight --time=$timestamp

done

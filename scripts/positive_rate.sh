export CUDA_VISIBLE_DEVICES=0

percent=100
# model=GPT4TS
model=PatchTST


for seq_len in 52
do
for pred_len in 8
do
for if_inverse in 0 3
do

python main.py \
    --root_path /data_disk/lichx/CQ_CDC/2010-2023流感数据/ \
    --data_path positive_rate_diff_pre.csv \
    --model_id flu+%_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 64 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 2 \
    --percent $percent \
    --gpt_layer 12 \
    --itr 3 \
    --model $model \
    --is_gpt 1 \
    --plt 0 \
    --read_model 0 \
    --write_model 0 \
    --features S \
    --target positive_rate \
    --if_inverse $if_inverse 
done
done
done

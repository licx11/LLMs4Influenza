#  nohup bash ./scripts/Gemma/positive_rate.sh > ./Output/PositiveRate/Gemma_leaky_relu.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

percent=100
model=Gemma2
train_epoch=64
pred_len=13

for seq_len in 52
do
for fix_seed in 2021 2024 42
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
    --batch_size 4 \
    --learning_rate 0.001 \
    --train_epochs $train_epoch \
    --decay_fac 0.75 \
    --d_model 3584 \
    --n_heads 4 \
    --d_ff 3584 \
    --freq 0 \
    --patch_size 16 \
    --stride 2 \
    --percent $percent \
    --itr 1 \
    --model $model \
    --is_gpt 1 \
    --plt 0 \
    --read_model 0 \
    --write_model 0 \
    --features S \
    --target positive_rate \
    --if_inverse $if_inverse \
    --fix_seed $fix_seed
    
done
done
done

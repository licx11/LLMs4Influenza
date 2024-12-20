#  nohup bash ./scripts/Llama/positive_rate.sh > ./Output/PositiveRate/Llama2_test.log 2>&1 &

#  nohup bash ./scripts/Llama/positive_rate.sh > ./Output/PositiveRate/Llama2_relu_lr4_predlen8.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

percent=100
llama_layer=32
model=Llama2
train_epoch=64
pred_len=8

for seq_len in 52
do
for fix_seed in 2021 2024 42
# for fix_seed in 42
do
for if_inverse in 0 3
# for if_inverse in 3
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
    --learning_rate 1e-4 \
    --train_epochs $train_epoch \
    --decay_fac 0.75 \
    --d_model 4096 \
    --n_heads 4 \
    --d_ff 4096 \
    --freq 0 \
    --patch_size 16 \
    --stride 2 \
    --percent $percent \
    --llama_layers $llama_layer \
    --itr 1 \
    --model $model \
    --is_gpt 1 \
    --plt 1 \
    --read_model 0 \
    --write_model 0 \
    --features S \
    --target positive_rate \
    --if_inverse $if_inverse \
    --fix_seed $fix_seed
    
done
done
done

export CUDA_VISIBLE_DEVICES=0

percent=100
llama_layer=6
model=Llama3
train_epoch=32
pred_len=13

for seq_len in 52
do
for fix_seed in 2021 2024 42
do
for if_inverse in 0 1
do

python main.py \
    --root_path /data_disk/lichx/CN_CDC/ \
    --data_path NorthChina_diff.csv \
    --model_id flu+%_in_NorthChina_$model'_'$llama_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 4 \
    --learning_rate 0.0001 \
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
    --plt 0 \
    --read_model 0 \
    --write_model 0 \
    --features S \
    --target positive_rate \
    --if_inverse $if_inverse \
    --order 1 \
    --fix_seed $fix_seed
done
done
done

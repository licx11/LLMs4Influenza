#  nohup bash ./scripts/Gemma/flucdc_south.sh > ./Output/SouthChinaFlu/Gemma_leaky_relu.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0

percent=100
model=Gemma2
train_epoch=64
pred_len=13

# for seq_len in 104 52
for seq_len in 52
do
for fix_seed in 2021 2024 42
do
for if_inverse in 0 1
# for if_inverse in  1
do

python main.py \
    --root_path /data_disk/lichx/CN_CDC/ \
    --data_path SouthChina_diff.csv \
    --model_id flu+%_in_SouthChina_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 4 \
    --learning_rate 0.001 \
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
    --order 0 \
    --fix_seed $fix_seed
done
done
done



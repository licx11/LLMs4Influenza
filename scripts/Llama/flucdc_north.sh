#  nohup bash ./scripts/Llama/flucdc_north.sh > ./Output/NorthChinaFlu/Llama2_relu_lr4_no_fc_no_freeze.log 2>&1 &

#  nohup bash ./scripts/Llama/flucdc_north.sh > ./Output/NorthChinaFlu/Llama2_test.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

percent=100
llama_layer=32
model=Llama2
train_epoch=64
# pred_len=13
seq_len=52

for pred_len in 13
do
echo "pred_len: $pred_len"
for fix_seed in 2021 2024 42
# for fix_seed in 2021
do
for if_inverse in 0 1
# for if_inverse in 1
do

python main.py \
    --root_path /data_disk/lichx/CN_CDC/ \
    --data_path NorthChina_diff.csv \
    --model_id flu+%_in_NorthChina_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
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

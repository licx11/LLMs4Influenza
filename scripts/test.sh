#  nohup bash ./scripts/test.sh > ./Output/test2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

percent=100
# model=GPT4TS
model=Llama2
pred_len=13

for seq_len in 52
do
for llama_layer in $(seq 8 15)
do
for if_inverse in 0 
do
printf "\nllama_layer: $llama_layer\n"
python main.py \
    --root_path /data_disk/lichx/all_datasets/ETT-small/ \
    --data_path ETTh2_5.csv \
    --model_id flu+%_$model'_'$llama_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 32 \
    --decay_fac 0.75 \
    --d_model 4096 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 2 \
    --percent $percent \
    --gpt_layer 6 \
    --llama_layers $llama_layer \
    --itr 1 \
    --model $model \
    --is_gpt 1 \
    --plt 0 \
    --read_model 0 \
    --write_model 0 \
    --features S \
    --target OT \
    --if_inverse $if_inverse \
    --order 0
done
done
done
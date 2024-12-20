export CUDA_VISIBLE_DEVICES=0

seq_len=336
# model=GPT4TS
model=Llama2

for percent in 100
do
for pred_len in 13
do
for lr in 0.001
do

python main.py \
    --root_path /data_disk/lichx/all_datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 32 \
    --decay_fac 0.5 \
    --d_model 4096 \
    --n_heads 4 \
    --d_ff 4096 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --write_model 1

done
done
done

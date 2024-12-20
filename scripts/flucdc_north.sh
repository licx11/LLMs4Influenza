export CUDA_VISIBLE_DEVICES=0

percent=100
# model=GPT4TS
model=PatchTST

gpt_layer=6
for seq_len in 52
do
for pred_len in 8
do
echo "pred_len: $pred_len"
for if_inverse in 0 1
do

python main.py \
    --root_path /data_disk/lichx/CN_CDC/ \
    --data_path NorthChina_diff.csv \
    --model_id NoTrain_flu+%_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
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
    --gpt_layers $gpt_layer \
    --itr 3 \
    --model $model \
    --is_gpt 1 \
    --plt 0 \
    --read_model 0 \
    --write_model 0 \
    --features S \
    --target positive_rate \
    --if_inverse $if_inverse \
    --order 1
done
done
done

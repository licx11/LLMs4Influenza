export CUDA_VISIBLE_DEVICES=0

percent=100
# model=GPT4TS
model=PatchTST
train_epoch=64

# for seq_len in 104 52; do
for seq_len in 52; do
    for pred_len in 8; do
        for if_inverse in 0 2; do
            python main.py \
                --root_path /data_disk/lichx/CQ_CDC/2010-2023流感数据/Weekly/ \
                --data_path Weekly_pre_diff.csv \
                --model_id Flu_Weekly_pre_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
                --data custom \
                --seq_len $seq_len \
                --label_len 18 \
                --pred_len $pred_len \
                --batch_size 4 \
                --lradj type4 \
                --learning_rate 0.0001 \
                --train_epochs $train_epoch \
                --decay_fac 0.75 \
                --d_model 768 \
                --n_heads 4 \
                --d_ff 768 \
                --gpt_layer 6 \
                --dropout 0.3 \
                --enc_in 7 \
                --c_out 7 \
                --freq 0 \
                --patch_size 16 \
                --stride 8 \
                --percent 100 \
                --itr 3 \
                --model $model \
                --tmax 20 \
                --cos 1 \
                --is_gpt 1 \
                --features S \
                --target num \
                --plt 0 \
                --if_inverse $if_inverse 
        done
    done
done


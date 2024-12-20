#  nohup bash ./scripts/Gemma/FLUw_pre.sh > ./Output/Weekly/Gemma_leaky_relu_6th.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0

percent=100
model=Gemma2
train_epoch=64
pred_len=13

# 4 8 12
# for seq_len in 104 52; do
for seq_len in 52; do
    # for fix_seed in 2021 2024 42; do
    for fix_seed in 42; do
        # echo "pred_len: $pred_len"
        for if_inverse in 2; do
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
                --learning_rate 0.001 \
                --train_epochs $train_epoch \
                --decay_fac 0.75 \
                --d_model 3584 \
                --n_heads 4 \
                --d_ff 3584 \
                --dropout 0.3 \
                --enc_in 7 \
                --c_out 7 \
                --freq 0 \
                --patch_size 16 \
                --stride 8 \
                --percent 100 \
                --itr 1 \
                --model $model \
                --tmax 20 \
                --cos 1 \
                --is_gpt 1 \
                --features S \
                --target num \
                --plt 0 \
                --if_inverse $if_inverse \
                --fix_seed $fix_seed
        done
    done
done


#  nohup bash ./scripts/Weekly/FLUw_city_pre.sh > ./Output/Weekly/output_FLU_cq_weekly_pre.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

# seq_len=336
# seq_len=48
seq_len=16
label_len=8

model=GPT4TS
# 4 8 12
for percent in 100; do
    # for pred_len in 8 12 16 20; do
    for pred_len in 4 8 12; do
        echo "pred_len: $pred_len"
        for lr in 0.0001; do
            python main.py \
                --root_path /data_disk/lichx/CQ_CDC/2010-2023流感数据/Weekly/ \
                --data_path Weekly_covid19_pre.csv \
                --model_id Flu_cq_Weekly_pre_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
                --data custom \
                --seq_len $seq_len \
                --label_len $label_len \
                --pred_len $pred_len \
                --batch_size 16 \
                --lradj type4 \
                --learning_rate $lr \
                --train_epochs 128 \
                --decay_fac 0.5 \
                --d_model 768 \
                --n_heads 4 \
                --d_ff 768 \
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
                --features S \
                --target num \
                --plt 1
        done
    done
done


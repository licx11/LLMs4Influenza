#  nohup bash ./scripts/Llama/test.sh > ./Output/Llama2_layers_last_1_to_6.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0

percent=100
# llama_layer=6
seq_len=52
model=Llama2


# for seq_len in 104 52
for llama_layer in $(seq 1 6)
do
    printf "\nllama_layer: $llama_layer\n"
    for pred_len in 13
    do
        for if_inverse in 0 
        do
            python main.py \
                --root_path /data_disk/lichx/all_datasets/ETT-small/ \
                --data_path ETTh2_5.csv \
                --model_id test_$model'_'$llama_layer'_'$seq_len'_'$pred_len'_'$percent \
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
                --d_ff 4096 \
                --freq 0 \
                --patch_size 16 \
                --stride 2 \
                --percent $percent \
                --llama_layers $llama_layer \
                --itr 3 \
                --model $model \
                --is_gpt 1 \
                --plt 0 \
                --read_model 0 \
                --write_model 0 \
                --features S \
                --target OT \
                --if_inverse $if_inverse 
        done
    done
done

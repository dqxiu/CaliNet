export WANDB_PROJECT=continual_pretrain
export MODEL=t5-base
lrs=(1e-4)
seeds=(1)
export fact_nums=1000
for lr in ${lrs[@]};do
  for seed in ${seeds[@]};do
  export EXPNAME=wq_facts_${lr}_vanilla_ft
  CUDA_VISIBLE_DEVICES=4 python run_kb_t5_nofreeze.py \
      --model_name_or_path ${MODEL} \
      --do_eval \
      --do_train \
      --do_predict \
      --lr_scheduler_type constant \
      --adafactor True \
      --train_file  /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_samedevtest/1ans_test256train128_train.csv \
      --validation_file /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_samedevtest/1ans_test256train128_val.csv \
      --test_file /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_samedevtest/1ans_test256train128_test.csv \
      --max_source_length 64 \
      --max_target_length 8 \
      --output_dir /mnt/data2/dqx/neural_kb_continue_pre/${EXPNAME} \
      --per_device_train_batch_size=1024 \
      --per_device_eval_batch_size=1024 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column src_sent \
      --learning_rate ${lr} \
      --seed ${seed} \
      --warmup_steps 100 \
      --ex_size 3072 \
      --kb_layer "" \
      --summary_column tgt_sent \
      --gradient_accumulation_steps 2 \
      --save_strategy steps \
      --num_train_epochs 1000 \
      --evaluation_strategy steps \
      --eval_steps 500 \
      --logging_steps 500 \
      --save_steps 500 \
      --logging_strategy steps \
      --max_steps 10000 \
      --report_to wandb \
      --run_name ${EXPNAME}
  done
done


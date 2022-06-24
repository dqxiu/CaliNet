export WANDB_PROJECT=probing_t5base
export MODEL=/mnt/data2/dqx/finetune_knowledge/100_facts_1e-3_vanilla_ft_t5-large
lrs=(1e-2)
seeds=(1)
export fact_nums=100
for lr in ${lrs[@]};do
  for seed in ${seeds[@]};do
  export EXPNAME=${fact_nums}_facts_test_t5-large_ori
  CUDA_VISIBLE_DEVICES=0 python run_kb_t5_freeze.py \
      --model_name_or_path ${MODEL} \
      --do_eval \
      --do_train false \
      --do_predict \
      --lr_scheduler_type constant \
      --adafactor True \
      --train_file  /home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_${fact_nums}/train.csv \
      --validation_file  /home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_${fact_nums}/false_test.csv \
      --test_file /home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_${fact_nums}/test.csv \
      --max_source_length 64 \
      --max_target_length 8 \
      --output_dir /mnt/data2/dqx/tmp/${EXPNAME} \
      --per_device_train_batch_size=512 \
      --per_device_eval_batch_size=512 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column src_sent \
      --learning_rate ${lr} \
      --seed ${seed} \
      --warmup_steps 100 \
      --summary_column tgt_sent \
      --gradient_accumulation_steps 4 \
      --save_strategy steps \
      --num_train_epochs 1000 \
      --evaluation_strategy epoch \
      --ex_size 512 \
      --kb_layer "" \
      --save_total_limit 1 \
      --logging_steps 100 \
      --save_steps 100 \
      --save_total_limit 1 \
      --logging_strategy steps \
      --max_steps 10000 \
      --report_to wandb \
      --run_name ${EXPNAME}
      # --fp16 \
      # --fp16_backend apex
  done
done


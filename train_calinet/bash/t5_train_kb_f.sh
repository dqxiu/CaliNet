export WANDB_PROJECT=layer_exp
export MODEL=t5-base
lrs=(1e-2)
dims=(64)
fact_nums=(100)
layers=(11)
for lr in ${lrs[@]};do
  for dim in ${dims[@]};do
    for fact_num in ${fact_nums[@]};do
      for layer in ${layers[@]};do
  export EXPNAME=${fact_num}_facts_layer${layer}
  CUDA_VISIBLE_DEVICES=2 python run_kb_t5_freeze.py \
      --model_name_or_path ${MODEL} \
      --do_eval \
      --do_train \
      --do_predict \
      --lr_scheduler_type constant \
      --adafactor True \
      --train_file  /home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_${fact_num}/train.csv \
      --validation_file  /home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_${fact_num}/val.csv \
      --test_file /home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_${fact_num}/test.csv \
      --max_source_length 64 \
      --max_target_length 8 \
      --output_dir /mnt/data2/dqx/neural_kb_result_layer/${EXPNAME} \
      --per_device_train_batch_size=512 \
      --per_device_eval_batch_size=512 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column src_sent \
      --learning_rate ${lr} \
      --seed 1 \
      --warmup_steps 100 \
      --summary_column tgt_sent \
      --gradient_accumulation_steps 4 \
      --save_strategy steps \
      --num_train_epochs 1000 \
      --evaluation_strategy epoch \
      --ex_size ${dim} \
      --kb_layer ${layer} \
      --save_total_limit 1 \
      --logging_steps 500 \
      --save_steps 500 \
      --save_total_limit 1 \
      --logging_strategy steps \
      --max_steps 5000 \
      --report_to wandb \
      --run_name ${EXPNAME}
        done
      done
  done
done


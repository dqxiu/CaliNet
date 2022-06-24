export WANDB_PROJECT=train_ffn_for_cbqa_overlap
export MODEL=t5-base
lrs=(1e-2)
seeds=(1)
fact_nums=(64)
datasets=(_test256train128_)
nsteps=(50000)
dims=(256)
for dim in ${dims[@]};do
  for lr in ${lrs[@]};do
    for fact_num in ${fact_nums[@]};do
      for dataset in ${datasets[@]};do
        for nstep in ${nsteps[@]};do
  export EXPNAME=fn_wq_1ans${dataset}${lr}_${nstep}_dim${dim}_tq
  CUDA_VISIBLE_DEVICES=2 python run_kb_t5_freeze.py \
      --model_name_or_path ${MODEL} \
      --do_eval \
      --do_train \
      --do_predict \
      --lr_scheduler_type constant \
      --adafactor True \
      --train_file  /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/1ans${dataset}train.csv \
      --validation_file  /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/1ans${dataset}val.csv \
      --test_file /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/1ans${dataset}test.csv \
      --max_source_length 64 \
      --max_target_length 8 \
      --output_dir /mnt/data2/dqx/neural_kb_result_overlap/${EXPNAME} \
      --per_device_train_batch_size=1024 \
      --per_device_eval_batch_size=1024 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column src_sent \
      --learning_rate ${lr} \
      --seed 1 \
      --warmup_steps 100 \
      --summary_column tgt_sent \
      --gradient_accumulation_steps 2 \
      --save_strategy steps \
      --evaluation_strategy steps \
      --ex_size ${dim} \
      --kb_layer 11 \
      --logging_steps 100 \
      --save_steps 1000 \
      --logging_strategy steps \
      --max_steps ${nstep} \
      --save_total_limit 1 \
      --report_to wandb \
      --run_name ${EXPNAME} \
      --load_best_model_at_end true
      done
  done
done
done
done


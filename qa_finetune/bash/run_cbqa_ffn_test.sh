export WANDB_PROJECT=t5base_cbqa_sup_meta_test
LRS=(4e-5)
DATA_NAMES=(128)
steps=(4000)
for LR in ${LRS[@]};do
for DATA_NAME in ${DATA_NAMES[@]};do
for step in ${steps[@]};do
export EXPNAME=no_ffn_nq_${DATA_NAME}_t5-base_${LR}_${step}
CUDA_VISIBLE_DEVICES=4 python3 cbqa/cbqa.py \
    --model_name_or_path /mnt/data2/dqx/neural_kb_continue_qa/no_ffn_wq_128_t5-base_1e-3_tq-1e-3 \
    --train_file /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/wq_add_k_test.json \
    --validation_file /home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/wq_add_k_test.json \
    --test_file /home/dqx/neural_kb/cbqa/data_previous/tq_dev.json \
    --dropout 0.2 \
    --kb_layer "" \
    --ex_size 64 \
    --do_train false\
    --do_eval \
    --do_predict \
    --metric_for_best_model exact_match \
    --question_column question \
    --answer_column answers \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 \
    --optim adafactor \
    --learning_rate ${LR} \
    --lr_scheduler_type constant \
    --max_seq_length 256 \
    --pad_to_max_length False \
    --output_dir /mnt/data2/dqx/neural_kb_result_newdim/result_ffn/${EXPNAME} \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --logging_strategy steps \
    --logging_steps 100 \
    --seed 1234  \
    --report_to wandb \
    --run_name ${EXPNAME} \
    --optim_group all \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --num_train_epochs 100 \
    --predict_with_generate \
    --load_best_model_at_end true
done
done
done
1. bash bash/run_cbqa.sh # finetune T5 
2. python gen_calidata/evaluate.py  --prediction_file finetuned_model_path/predictions.json  --gen true
3. python gen_calidata/get_entities.py (conda activate srl)
4. python ../train_calinet/dataset/pararel/preprocess_for_t5_entpair_index.py
5. python ../train_calinet/dataset/pararel/add_patterns.py
6. python gen_calidata/reaggre_for_ffn.py

Following the steps above, data for calibration are generated. Then use the code in "gen_calidata/" to calibrate fraud knowledge for QA.


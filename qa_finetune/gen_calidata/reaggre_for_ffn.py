# import argparse
# import glob
# import pickle
# import random
# import os
# import csv
# import torch
# from tqdm import tqdm
# import json


# with open("/home/dqx/neural_kb/cbqa/data/wq_train_128.json",'r') as load_f:
#     ori_train_dict = json.load(load_f)
# # the code is for 1ans
# with open("/home/dqx/neural_kb/cbqa/gen_unlearned/no_ffn_wq_128_t5-base_4e-5/trex_output.json",'r') as load_f:
#     ori_test_dict = json.load(load_f)


# new_train_dict={'data':[]}
# new_test_dict={'data':[]}

# ans_list=[]
# with open("/home/dqx/neural_kb/cbqa/gen_unlearned/no_ffn_wq_128_t5-base_4e-5/trex_output.json",'r') as load_f:
#     trex_dict = json.load(load_f)
#     for id_key in trex_dict.keys():
#         example = trex_dict[id_key]
#     for ans in example['answers']:
#         ans_list.append(ans)
# with open("/home/dqx/neural_kb/cbqa/wq_split_dev.json",'r') as load_f_dev:
#     load_dev_dict = json.load(load_f_dev)
#     data_dev = load_dev_dict['data']
#     for exp_dict in data_dev:
#         exp_ans = exp_dict["answers"]
#         for ans in exp_ans:
#             if ans not in ans_list:
#                 ans_list.append(ans)
# new_examples=[]
# id_list=[]
# cnt=0
# for id_key in trex_dict.keys():
#     example = trex_dict[id_key]
#     example_trex=example['trex'][0]
#     if example_trex['sub_label'] not in ans_list and example_trex['obj_label'] not in ans_list:
#         if cnt<28:
#             example['id']=int(id_key)
#             id_list.append(int(id_key))
#             new_examples.append(example)
#             cnt+=1

# for exa in ori_train_dict['data']:
#     if cnt<128:
#         new_examples.append(exa)
#         id_list.append(exa['id'])
#         cnt+=1
# random.shuffle(new_examples)
# new_train_dict['data']=new_examples

# for id_key in ori_test_dict.keys():
#     example = trex_dict[id_key]
#     if int(id_key) not in id_list:
#         example['id']=int(id_key)
#         new_test_dict['data'].append(example)
# print(len(new_test_dict['data']))
# with open("/home/dqx/neural_kb/cbqa/data/wq_train100+28.json", 'w') as write_f:
# 	json.dump(new_train_dict, write_f, indent=4, ensure_ascii=False)
# with open("/home/dqx/neural_kb/cbqa/data/wq_test_max-28.json", 'w') as write_f:
# 	json.dump(new_test_dict, write_f, indent=4, ensure_ascii=False)

import argparse
import glob
import pickle
import random
import os
import csv
import torch
from tqdm import tqdm
import json


with open("/home/dqx/neural_kb/cbqa/data_previous/tq_train.json",'r') as load_f:
    ori_train_dict = json.load(load_f)
# the code is for 1ans
with open("/home/dqx/neural_kb/cbqa/data_previous/tq_dev.json",'r') as load_f:
    ori_dev_dict = json.load(load_f)

new_train_dict={'data':[]}
new_add_k_test_dict={'data':[]}
new_no_k_test_dict={'data':[]}

unlearned_lst=[]
with open("/home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/trex_output.json",'r') as load_f:
    trex_dict = json.load(load_f)
    add_k_id=[trex_dict[key]["question"] for key in trex_dict.keys()]
print(len(add_k_id))

with open("/home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/unlearned_all_entpairs.json",'r') as load_f:
    unlearned_dict = json.load(load_f)
    for item in unlearned_dict['data']:
        unlearned_lst.append(item['question'])


for item in ori_dev_dict['data']:
    if item["question"] in unlearned_lst and item['question'] in add_k_id:
        new_add_k_test_dict['data'].append(item)
    else:
        new_no_k_test_dict['data'].append(item)


print(len(new_add_k_test_dict['data']))
print(len(new_no_k_test_dict['data']))
with open("/home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/wq_add_k_test.json", 'w') as write_f:
	json.dump(new_add_k_test_dict, write_f, indent=4, ensure_ascii=False)
with open("/home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/wq_add_no_k_test.json", 'w') as write_f:
	json.dump(new_no_k_test_dict, write_f, indent=4, ensure_ascii=False)
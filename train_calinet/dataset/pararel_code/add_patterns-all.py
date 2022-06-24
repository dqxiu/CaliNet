import argparse
import glob
import pickle
import random
import os
import csv
from pararel.consistency import utils
import torch
from tqdm import tqdm
import json

with open("/home/dqx/neural_kb/cbqa/gen_unlearned/no_ffn_wq_128_t5-base_4e-5/trex_output.json",'r') as load_f:
    Details = ['src_sent', 'tgt_sent']
    output_path='/home/dqx/neural_kb/cbqa/gen_unlearned/no_ffn_wq_128_t5-base_4e-5/all_ans_'
    with open(output_path+"test.csv","w",encoding="utf-8") as csvfile_s:
        with open(output_path+"train.csv","w",encoding="utf-8") as csvfile_t:
            with open(output_path+"val.csv","w",encoding="utf-8") as csvfile_v:
                writer_t = csv.writer(csvfile_t)
                writer_t.writerow(Details)
                writer_v = csv.writer(csvfile_v)
                writer_v.writerow(Details)
                writer_s = csv.writer(csvfile_s)
                writer_s.writerow(Details)
                entity_lst = []
                entity_test_lst = []
                entity_val_lst = []

                graph_path = "data/pattern_data/graphs/"
                load_dict = json.load(load_f)
                for id_key in load_dict.keys():
                    # [0] for single_ans
                    relation_list=[]
                    for i,triplet in enumerate(load_dict[id_key]['trex']):

                        relation=triplet['relation']
                        if relation in relation_list:
                            break
                        relation_list.append(relation)
                        d=triplet
                        relation_path = graph_path +relation+".graph"
                        if relation in ["P1376", "P36"]:
                            print("1-1 relation")
                            rel_type=0
                        elif relation in ["P166", "P69", "P47", "P463", "P101", "P1923", "P106", "P527", "P530", "P27", "P178", "P1412", "P108", "P39", "P937", "P1303", "P190", "P1001", "P31"]:
                            print("N-M relation")
                            rel_type=1
                        else:
                            print("N-1 relationship")
                            rel_type=2
                        

                        with open(relation_path, "rb") as f:
                            graph = pickle.load(f)
                        node_split = 0 
                        for node in graph.nodes():
                            node_split += 1
                            pattern = node.lm_pattern
                            if pattern.find("[X]") < pattern.find("[Y]"):
                                subj_start_id_char = pattern.find("[X]")
                                subj_end_id_char = subj_start_id_char + len(d["sub_label"])
                                obj_start_id_char = pattern.find("[Y]")
                                obj_start_id_char = obj_start_id_char + len(d["sub_label"]) - 3
                                obj_end_id_char = obj_start_id_char + len(d["obj_label"])
                            else:
                                obj_start_id_char = pattern.find("[Y]")
                                obj_end_id_char = obj_start_id_char + len(d["obj_label"])
                                subj_start_id_char = pattern.find("[X]")
                                subj_start_id_char = subj_start_id_char + len(d["obj_label"]) - 3
                                subj_end_id_char = subj_start_id_char + len(d["sub_label"])
                            
                            if rel_type==0:
                                entity_p = torch.rand((1,1))
                                if entity_p < 0.5:
                                    src_sent = pattern.replace("[X]", d["sub_label"]).replace("[Y]", "<extra_id_0>")
                                    tgt_sent = "<extra_id_0> " + d["obj_label"] + " <extra_id_1>"
                                else:
                                    src_sent = pattern.replace("[Y]", d["obj_label"]).replace("[X]", "<extra_id_0>")
                                    tgt_sent = "<extra_id_0> " + d["sub_label"] + " <extra_id_1>"
                            else:
                                src_sent = pattern.replace("[X]", d["sub_label"]).replace("[Y]", "<extra_id_0>")
                                tgt_sent = "<extra_id_0> " + d["obj_label"] + " <extra_id_1>"

                            if node_split%8 == 1:
                                entity_test_lst.append([src_sent, tgt_sent])
                            elif node_split%8 == 2:
                                entity_val_lst.append([src_sent, tgt_sent])
                            else:
                                entity_lst.append([src_sent, tgt_sent])
                entity_lst=list(set([tuple(t) for t in entity_lst]))
                entity_test_lst=list(set([tuple(t) for t in entity_test_lst]))
                entity_val_lst=list(set([tuple(t) for t in entity_val_lst]))
                writer_t.writerows(entity_lst)
                writer_s.writerows(entity_test_lst)
                writer_v.writerows(entity_val_lst)




# with open("/home/dqx/neural_kb/fact_checker/dataset/pararel/wq_datagen_entpair/", 'w') as write_f:
# 	json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
# print(len(save_dict.keys()))
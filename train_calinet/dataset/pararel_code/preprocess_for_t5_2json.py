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


def generate_data(num_relations, num_tuples,num_total_facts, relations_given, LAMA_path):

    graph_path = "data/pattern_data/graphs/"
    relations_path = glob.glob(graph_path + "*.graph")
    output_path = "probing_data_"
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    save_dict={"data":[]}

    random.shuffle(relations_path)
    relation_path_keep = []
    metadata = "_"
    if relations_given != "":
        relations_given = sorted(relations_given.split(","))
        for relation_path in relations_path:
            relation = relation_path.split("/")[-1].split(".")[0]
            if relation in relations_given:
                relation_path_keep.append(relation_path)
        metadata += "_".join(relations_given)
        metadata += "-"
    if len(relation_path_keep) < num_relations:
        for relation_path in relations_path:
            if relation_path not in relation_path_keep:
                relation = relation_path.split("/")[-1].split(".")[0]
                relation_path_keep.append(relation_path)
                metadata += relation
                metadata += "-"
                if len(relation_path_keep) == num_relations:
                    break
    # metadata = metadata.strip("-")
    # metadata=""
    # output_path = output_path + str(num_tuples) + "_" + str(num_relations) + metadata + "/"
    # Details = ['src_sent', 'tgt_sent']
    # os.mkdir(output_path)
    # with open(output_path+"test.csv","a",encoding="utf-8") as csvfile_s:
    #     with open(output_path+"train.csv","a",encoding="utf-8") as csvfile_t:
    #         with open(output_path+"val.csv","a",encoding="utf-8") as csvfile_v:
    #             writer_t = csv.writer(csvfile_t)
    #             writer_t.writerow(Details)
    #             writer_v = csv.writer(csvfile_v)
    #             writer_v.writerow(Details)
    #             writer_s = csv.writer(csvfile_s)
    #             writer_s.writerow(Details)
        
                # if not os.path.exists(output_path):
    print("Saving data to: ", output_path)
    

    # output_path_true = output_path + "train_"
    # output_path_mlm = output_path + "train_mlm.txt"
    # f_mlm = open(output_path_mlm, "w")
    
    cnt_total_facts=0
    for relation_path in relation_path_keep:

        entity_lst = []
        entity_test_lst = []
        entity_val_lst = []
        
        try:
            with open(relation_path, "rb") as f:
                graph = pickle.load(f)
        except:
            continue
        print(relation_path)
        relation = relation_path.split("/")[-1].split(".")[0]


        # f_true = open(output_path_true + relation + ".txt", "w")
        print(relation)


        if relation in ["P1376", "P36"]:
            print("1-1 relation")
            rel_type=0

        elif relation in ["P166", "P69", "P47", "P463", "P101", "P1923", "P106", "P527", "P530", "P27", "P178", "P1412", "P108", "P39", "P937", "P1303", "P190", "P1001", "P31"]:
            print("N-M relation")
            rel_type=1

        else:
            print("N-1 relationship")
            print(relation)
            rel_type=2


        data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")
        random.shuffle(data)

        node_num = 0
        for node in graph.nodes():
            node_num+=1
        # if node_num<5:
        if node_num<5:
            continue
        print(len(data))
        # print(data[0])
        # input()

        data_clean =[]
        for item in tqdm(data):
            if item["sub_label"] in ["", ",", " ", "he", "She", "He", "she", "It", "it"] or item["obj_label"] in ["", ",", " ", "it", "It", "he", "She", "He", "she"]:
                continue
            result_s = all(c.isalnum() or c.isspace() or c in ["-", ","] for c in item["sub_label"])
            result_o = all(c.isalnum() or c.isspace() or c in ["-", ","] for c in item["obj_label"])
            if not(result_s or result_o):
                continue
            if not item in data_clean:
                data_clean.append(item)	

        print(len(data_clean))
        print(data_clean[0])
        # input()
        valid_num=0
        sub2obj=dict()
        obj2sub=dict()
        for i, d in tqdm(enumerate(data_clean)):
            if d["sub_label"] not in sub2obj.keys():
                sub2obj[d["sub_label"]]=[]
            sub2obj[d["sub_label"]].append(d["obj_label"])
        for i, d in tqdm(enumerate(data_clean)):
            if d["obj_label"] not in obj2sub.keys():
                obj2sub[d["obj_label"]]=[]
            obj2sub[d["obj_label"]].append(d["sub_label"])
        for i, d in tqdm(enumerate(data_clean)):
            valid_num+=1
            random.shuffle(data)
            node_split = 0 
            cnt_total_facts+=1
            cur_sentences=[]
            save_dict["data"].append({"fact_id":cnt_total_facts,"relation":relation,"triplet":d,"sentences":cur_sentences})
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
                

                entity_p = torch.rand((1,1))
                if entity_p < 0.5:
                    src_sent = pattern.replace("[X]", d["sub_label"]).replace("[Y]", "<extra_id_0>")
                    tgt_sent = "<extra_id_0> " + d["obj_label"] + " <extra_id_1>"
                    for sample_obj in random.sample(list(set(obj2sub.keys())),len(list(set(obj2sub.keys())))):
                        if sample_obj not in sub2obj[d["sub_label"]]:
                            false_ent=sample_obj
                            break
                    false_tgt_sent="<extra_id_0> " + false_ent + " <extra_id_1>"
                else:
                    src_sent = pattern.replace("[Y]", d["obj_label"]).replace("[X]", "<extra_id_0>")
                    tgt_sent = "<extra_id_0> " + d["sub_label"] + " <extra_id_1>"
                    for sample_sub in random.sample(list(set(sub2obj.keys())),len(list(set(sub2obj.keys())))):
                        if sample_sub not in obj2sub[d["obj_label"]]:
                            false_ent=sample_sub
                            break
                    false_tgt_sent="<extra_id_0> " + false_ent + " <extra_id_1>"

                
                cur_sentences.append([src_sent,tgt_sent,false_tgt_sent])
                

                # else:
                #     if pattern.find("[X]") < pattern.find("[Y]"):
                #         tgt_sent = "<extra_id_0> " + d["sub_label"] + " <extra_id_1> " + d["obj_label"] + " <extra_id_2>"
                #     else:
                #         tgt_sent = "<extra_id_0> " + d["obj_label"] + " <extra_id_1> " + d["sub_label"] + " <extra_id_2>"
                #     src_sent = pattern.replace("[X]", "<extra_id_0>").replace("[Y]", "<extra_id_1>")

                # pattern = pattern.replace("[X]", d["sub_label"])
                # pattern = pattern.replace("[Y]", "[MASK]")
                # pattern_mlm = pattern.replace("[MASK]", d["obj_label"])

                # f_true.write(src_sent)
                # f_true.write("\n")
                # f_mlm.write(pattern_mlm)
                # f_mlm.write("\n")

                # subj_start_id_char = pattern_mlm.find(d["sub_label"])
                # subj_end_id_char = subj_start_id_char + len(d["sub_label"])
                # obj_start_id_char = pattern_mlm.find(d["obj_label"])
                # obj_end_id_char = obj_start_id_char + len(d["obj_label"])
                
                # if node_split%10 == 8:
                #     entity_test_lst.append([src_sent, tgt_sent])
                # elif node_split%10 == 9:
                #     entity_val_lst.append([src_sent, tgt_sent])
                # else:
                #     entity_lst.append([src_sent, tgt_sent])

            # f_true.write("\n")


            if valid_num >= num_tuples:
                break
        # if cnt_total_facts >= num_total_facts:
        #         break
        with open(output_path+"trex_500each.json", 'w') as write_f:
	        json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
    #     f_true.close()
    # f_mlm.close()


        # entity_lst=list(set([tuple(t) for t in entity_lst]))
        # entity_test_lst=list(set([tuple(t) for t in entity_test_lst]))
        # entity_val_lst=list(set([tuple(t) for t in entity_val_lst]))
        # writer_t.writerows(entity_lst)
        # writer_s.writerows(entity_test_lst)
        # writer_v.writerows(entity_val_lst)
        # print(f"total facts: {cnt_total_facts}")
              



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations', '-nr', type=int, default=41, help='number of relations')
    parser.add_argument('--num_tuples', '-nt', type=int, default=500, help='number of tuples')
    parser.add_argument('--num_total_facts', '-nf', type=int, default=1000, help='number of tuples')
    parser.add_argument('--relations_given', '-r', type=str, default="", help='which relations')
    parser.add_argument('--LAMA_path', '-pair', type=str,
                        default="/home/dqx/neural_kb/fact_checker/dataset/trex/cleaned_T_REx/", help='number of tuples')

    args = parser.parse_args()


    generate_data(args.num_relations, args.num_tuples,args.num_total_facts,
                  args.relations_given, args.LAMA_path)


if __name__ == "__main__":
    main()

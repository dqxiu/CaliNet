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
import jsonlines


def generate_data(num_relations, num_tuples,num_total_facts, relations_given, LAMA_path):
    temp_dict=dict()
    output_id=0
    with jsonlines.open('relations.jsonl', 'r') as reader:
        for row in reader:
            temp_dict[row["relation"]]=row
    output_dict=dict()
    graph_path = "data/pattern_data/graphs/"
    relations_path = glob.glob(graph_path + "*.graph")
    output_path = "assess_"

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
    metadata=""
    output_path = output_path + str(num_tuples) + "_" + str(num_relations) + metadata + "/"
    os.mkdir(output_path)
    print("Saving data to: ", output_path)

    
    cnt_total_facts=0
    for relation_path in relation_path_keep:

        relation = relation_path.split("/")[-1].split(".")[0]
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

        data_clean =[]
        for item in tqdm(data):
            if item["sub_label"] in ["", ",", " ", "he", "She", "He", "she", "It", "it"] or item["obj_label"] in ["", ",", " ", "it", "It", "he", "She", "He", "she"]:
                continue
            result_s = all(c.isalnum() or c.isspace() or c in ["-", ","] for c in item["sub_label"])
            result_o = all(c.isalnum() or c.isspace() or c in ["-", ","] for c in item["obj_label"])
            if not(result_s or result_o):
                continue
            if len(data_clean)==num_tuples:
                break
            if not item in data_clean:
                data_clean.append(item)	

        print(len(data_clean))
        print(data_clean[0])

        temps=temp_dict[relation]
        valid_num=0
        for i, d in tqdm(enumerate(data_clean)):
            # print(temps)
            pattern = temps['template']
            src_sent = pattern.replace("[X]", d["sub_label"]).replace("[Y]", "<extra_id_0>")
            tgt_sent =d["obj_label"]
            output_dict[str(output_id)]={"true":(src_sent,tgt_sent),"false":[],"relation":""}
            output_dict[str(output_id)]["relation"]=relation
            for pattern in temps['wrong']:
                src_sent = pattern.replace("[X]", d["sub_label"]).replace("[Y]", "<extra_id_0>")
                tgt_sent =d["obj_label"]
                output_dict[str(output_id)]["false"].append((src_sent,tgt_sent))
            output_id+=1
            
            

            
    return output_dict,output_path
            




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations', '-nr', type=int, default=46, help='number of relations')
    parser.add_argument('--num_tuples', '-nt', type=int, default=1000, help='number of tuples')
    parser.add_argument('--num_total_facts', '-nf', type=int, default=4600, help='number of tuples')
    parser.add_argument('--relations_given', '-r', type=str, default="P264", help='which relations')
    parser.add_argument('--LAMA_path', '-pair', type=str,
                        default="/home/dqx/neural_kb/fact_checker/dataset/trex/cleaned_T_REx/", help='number of tuples')

    args = parser.parse_args()


    output_dict,output_path=generate_data(args.num_relations, args.num_tuples,args.num_total_facts,
                  args.relations_given, args.LAMA_path)
    with open(output_path+"assessing_data.json", 'w') as write_f:
	    json.dump(output_dict, write_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
import csv
sys.path.append("../../..")
import pickle
import json
import warnings
import torch.nn.functional as F
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, T5Tokenizer
# , T5ForConditionalGeneration
from src.modeling_t5 import T5ForConditionalGeneration
# from transformers import T5ForConditionalGeneration as T5ForConditionalGenerationOri
from transformers.generation_utils import GenerationMixin

# warnings.filterwarnings('ignore')


def probe_t5(model,input_ids, target):
    # print('Logits probing:')
    # labels = tokenizer.encode("French's border country is <extra_id_0>", return_tensors='pt')
    # outputs = model(input_ids=input_ids, labels=input_ids, output_hidden_states=True, return_dict=True)
    outputs = model(input_ids=input_ids, decoder_input_ids=torch.tensor([[0, 32099]],device='cuda:0'), output_hidden_states=True, return_dict=True)
    

    logits = outputs['logits'][0, -1]

    probs = F.softmax(logits, dim=-1)

    probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

    probs_ = []

    for index, prob in enumerate(probs):
        probs_.append((index, prob))
    # rank = sorted(probs_, key=lambda x: x[1], reverse=True)[:]
    # for idx,t in enumerate(rank):
    #     if t[0]==target.item():
    #         cur_rank=idx
    #         continue

    # rank_dict = [(t[1].item(), tokenizer.decode(t[0])) for t in rank]
    all_dict = dict()

    all_score=0
    for t in probs_:
        all_dict[t[0]]=t[1].item()
    # print(f"score: {all_dict[target.item()]}")
    # print(f"rank: {cur_rank}")
    return all_dict[target.item()]


if __name__ == '__main__':
    with torch.no_grad():

        # model_name = 't5-large'
        # # model_name = 't5-large'
        # model_config = AutoConfig.from_pretrained(
        #     model_name,
        #     revision="main",
        #     use_auth_token=False
        # )
        # device = 'cuda:0'
        # tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # model = T5ForConditionalGeneration.from_pretrained(model_name, config=model_config).to(device)
        # model.eval()
        device='cuda:0'
        model_config = AutoConfig.from_pretrained(
            "${OUTPUT_PATH}/finetune_knowledge/100_facts_1e-3_vanilla_ft_t5-large",
            revision="main",
            use_auth_token=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "${OUTPUT_PATH}/finetune_knowledge/100_facts_1e-3_vanilla_ft_t5-large",
            use_fast=True,
            revision="main",
            use_auth_token=False,
        )
        model_config.ex_size = 256
        model_config.kb_layer = ""
        model = T5ForConditionalGeneration.from_pretrained("${OUTPUT_PATH}/finetune_knowledge/100_facts_1e-3_vanilla_ft_t5-large", config=model_config).to(device)
        model.eval()
        
        path="${PROJECT_PATH}/train_calinet/dataset/pararel/probing_data_100/"
        with open(path+"assessing_prob100.json",'r') as load_f:
            load_dict = json.load(load_f)

        
        save_dict={"data":[]}
        batch_true = []
        batch_tgt = []
        batch_false = []
        for key in tqdm(load_dict.keys()):
            src_true = load_dict[key]["true"][0]
            target_token = load_dict[key]["true"][1]

            target = tokenizer.encode(target_token, return_tensors="pt").to(device)[0][0]
            
            # print(src_true)
            input_ids = tokenizer.encode(src_true, return_tensors="pt").to(device)
            P_true = probe_t5(model,input_ids, target) 
            P_false=0
            for false_tuple in load_dict[key]["false"]:
                # print(false_tuple[0])
                input_ids = tokenizer.encode(false_tuple[0], return_tensors="pt").to(device)
                P_false += probe_t5(model,input_ids, target) 
            P_false/=3
            
            cur_dict={"id":key, "src_true":src_true, "target_token":target_token, "false_tuple":load_dict[key]["false"],"P_true":P_true,"P_false":P_false,"CKA":P_true/P_false, "relation":load_dict[key]["relation"]}
            # if int(key)>5:
            #     break
            save_dict["data"].append(cur_dict)
        with open(path+"t5_large_cp_score.json", 'w') as write_f:
	        json.dump(save_dict, write_f, indent=4, ensure_ascii=False)


        





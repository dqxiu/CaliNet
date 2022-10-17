import argparse
import json
import os
import sys

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
import csv

def probe_t5(model,input_ids, target):

    outputs = model(input_ids=input_ids, decoder_input_ids=torch.tensor([[0, 32099]],device='cuda:0'), output_hidden_states=True, return_dict=True)
    

    logits = outputs['logits'][0, -1]

    probs = F.softmax(logits, dim=-1)

    probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

    probs_ = []

    for index, prob in enumerate(probs):
        probs_.append((index, prob))

    all_dict = dict()

    all_score=0
    for t in probs_:
        all_dict[t[0]]=t[1].item()

    return all_dict[target.item()]


if __name__ == '__main__':
    with torch.no_grad():


        device='cuda:0'
        model_config = AutoConfig.from_pretrained(
            "t5-large",
            revision="main",
            use_auth_token=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "t5-large",
            use_fast=True,
            revision="main",
            use_auth_token=False,
        )
        model_config.ex_size = 256
        model_config.kb_layer = ""
        model = T5ForConditionalGeneration.from_pretrained("t5-large", config=model_config).to(device)
        model.eval()

        all_data = []
        with open("cka/manual_test.csv") as csvfile:
            csv_reader = csv.reader(csvfile)  
            for row in csv_reader:  
                if len(row)==5:
                    all_data.append(row)
            print(len(all_data))
        
        cnt = 0
        for sent_id in tqdm(range(len(all_data))):
            src = all_data[sent_id][0]
            true_target = all_data[sent_id][1]
            false_targets = all_data[sent_id][2:5]


            input_ids = tokenizer.encode(src, return_tensors="pt").to(device)
            true_target = true_target.strip().strip('.').strip('<extra_id_0>').strip('<extra_id_1>').strip()
            true_target_input_ids = tokenizer.encode(true_target, return_tensors="pt").to(device)[0][0]
            
 
            P_true = probe_t5(model,input_ids, true_target_input_ids) 
            P_false = 0
            for false_tuple in false_targets:
                # print(false_tuple)
                false_tuple = false_tuple.strip().strip('.').strip('<extra_id_0>').strip('<extra_id_1>').strip()
                # print(false_tuple)
                false_target_input_ids = tokenizer.encode(false_tuple, return_tensors="pt").to(device)[0][0]
                P_false += probe_t5(model,input_ids, false_target_input_ids) 
            P_false/=3
            print(f"{P_true} {P_false}")
            if (P_true)<(P_false):
                cnt+=1
            
        print(cnt/50) 



        





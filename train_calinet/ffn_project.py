import argparse
import json
import pickle
import warnings
import os
import sys
sys.path.append("..")
import _jsonnet
import numpy as np
import pyhocon
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer,T5Tokenizer, T5ForConditionalGeneration
from src.modeling_t5 import T5ForConditionalGeneration as T5ForConditionalGenerationEx
warnings.filterwarnings('ignore')





# Generating WikiLM Logits
def get_all_projected_values(model, E, mode):
    values = []
    if mode=="ori":
        layer_fc2_vals = [
            # model[f"decoder.layers.{layer_i}.fc2.weight"].T decoder.block.11.layer.2.DenseReluDense.wo_ex
            model.decoder.block[layer_i].layer[2].DenseReluDense.wo.weight.T
            for layer_i in range(12)
        ]
        for layer in range(12):
            for dim in range(3072):
                values.append(layer_fc2_vals[layer][dim].unsqueeze(0))
    if mode=="ex_ffn":
        for dim in range(3072):
            values.append(model.decoder.block[11].layer[2].DenseReluDense.wo_ex.weight.T[dim].unsqueeze(0))
    values = torch.cat(values)
    logits = E.matmul(values.T).T.detach()
    return logits

def get_vocab(logits, mode):
    # Projecting the Logits Over the Vocabulary
    top_k = 10
    cnt=0
    projections = []
    d=dict()
    inv_d=dict()
    projections = {}
    if mode=="ori":
        for i in range(12):
            for j in range(3072):
                d[cnt] = (i, j)
                inv_d[(i, j)] = cnt
                cnt += 1
        for i in range(12):
            for j in range(3072):
                k = (i, j)
                cnt = inv_d[(i, j)]
                ids = np.argsort(-logits[cnt])[:top_k].tolist()
                tokens = [tokenizer._convert_id_to_token(x) for x in ids]
                projections[k] = [tokens[b] for b in range(len(tokens))]
    if mode=="ex_ffn":
        i=12
        for j in range(3072):
            d[cnt] = (i, j)
            inv_d[(i, j)] = cnt
            cnt += 1
        for j in range(3072):
            k = (i, j)
            cnt = inv_d[(i, j)]
            ids = np.argsort(-logits[cnt])[:top_k].tolist()
            tokens = [tokenizer._convert_id_to_token(x) for x in ids]
            projections[k] = [tokens[b] for b in range(len(tokens))]
    return projections,d

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def get_predicted_clusters(n, cosine_mat):
    clustering = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='complete')
    predicted = clustering.fit(cosine_mat)
    predicted_clusters = predicted.labels_
    return predicted_clusters

if __name__ == '__main__':

    config = AutoConfig.from_pretrained(
        "/mnt/data2/dqx/neural_kb_result/1000_facts",
        revision="main",
        use_auth_token=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/data2/dqx/neural_kb_result/1000_facts",
    use_fast=True,
    revision="main",
    use_auth_token=False,
    )
    device = 'cuda'
    tokenizer = T5Tokenizer.from_pretrained("t5-base")




    config.ex_size = 3072
    config.kb_layer = "11"
    modelEx = T5ForConditionalGenerationEx.from_pretrained("/mnt/data2/dqx/neural_kb_result/1000_facts_longtime2",config=config).to(device)
    logits=get_all_projected_values(modelEx, modelEx.shared.weight, mode="ex_ffn")
    projections,d =get_vocab(logits.cpu().numpy(), mode="ex_ffn")
    
    print(projections)

    num_clusters=1000
    cosine_mat = cosine_distance_torch(logits).detach().cpu().numpy()
    predicted_clusters = get_predicted_clusters(num_clusters, cosine_mat)
    clusters = {i: [] for i in range(num_clusters)}
    for i, x in enumerate(predicted_clusters):
        clusters[x].append(d[i])
    clusters = {i: [] for i in range(num_clusters)}
    for i, x in enumerate(predicted_clusters):
        clusters[x].append(d[i])
    inv_map = {vi: k for k, v in clusters.items() for vi in v}


    

                          
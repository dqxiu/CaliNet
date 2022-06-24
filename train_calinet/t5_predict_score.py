import re

""" Official evaluation script for v1.1 of the SQuAD dataset. """

import argparse
import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    pattern = r',"*<extra_id_0> (.*) <extra_id_1>'
    for item_id in range(len(predictions)):
        total+=1
        ans=re.search(pattern, dataset[item_id], re.M|re.I)
        if ans==None:
            print("None")
            continue

        ground_truths = [ans.group(1).strip()]
        print(ground_truths)
        
        
        prediction = predictions[item_id]
        print(prediction)
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}



if __name__ == "__main__":
    expected_version = "1.1"
    # parser = argparse.ArgumentParser(description="Evaluation for SQuAD " + expected_version)
    # parser.add_argument("dataset_file", help="Dataset file")
    # parser.add_argument("prediction_file", help="Prediction File")
    # args = parser.parse_args()
    # with open(dataset_file) as dataset_file:
    #     dataset_json = json.load(dataset_file)
    #     if dataset_json["version"] != expected_version:
    #         print(
    #             "Evaluation expects v-" + expected_version + ", but got dataset with v-" + dataset_json["version"],
    #             file=sys.stderr,
    #         )
    #     dataset = dataset_json["data"]
    # with open(prediction_file) as prediction_file:
    #     predictions 0= json.load(prediction_file)
    fact_nums=100

    preds = open(f"/mnt/data2/dqx/finetune_knowledge/100_facts_1e-3_vanilla_ft_t5-base/generated_predictions.txt", "r").read().splitlines()
    refs = open(f"/home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_{fact_nums}/test.csv", "r").read().splitlines()[1:]
    dataset=refs
    predictions=preds
    
    for idx in range(len(preds)):
        preds[idx]=preds[idx].strip()
    predictions=preds
    print(json.dumps(evaluate(dataset, predictions)))
    print(len(preds))

# /mnt/data2/dqx/neural_kb_result_dim/100_facts_dim16/generated_predictions.txt
# preds = open("/mnt/data2/dqx/neural_kb_result/2500_facts/generated_predictions.txt", "r").read().splitlines()
# for idx in range(len(preds)):
#     preds[idx]=preds[idx].strip()

# cnt=0
# em=0
# cmp=list()
# refs = open("/home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_2500/test.csv", "r").read().splitlines()[1:]
# pattern = r',"*<extra_id_0> (.*) <extra_id_1>'
# for idx in range(len(refs)):
#     ans=re.search(pattern, refs[idx], re.M|re.I)
#     # if ans != None:
#     #     continue
#     refs[idx]=ans.group(1).strip()
#     if refs[idx]==preds[idx]:
#         em+=1
#     if preds[idx].lower() in refs[idx].lower() or refs[idx].lower() in preds[idx].lower():
#         cnt+=1
#     # else:
#     #     print(ref)
#     cmp.append(str(refs[idx])+", "+str(preds[idx])+"\n")
# if len(preds) == len(refs):
#     print(cnt/len(preds))
#     print(em/len(preds))
# with open("cmp.txt",'w') as fout:
#     for cmp_line in cmp:
#         fout.write(cmp_line)

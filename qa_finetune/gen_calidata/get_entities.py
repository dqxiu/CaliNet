import json
import argparse
import spacy
import random
spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")
calibrate_ents=[]
all_unlearned_dicts=[]
unlearned_entpair_dict={'data':[]}
learned_entpair_dict={'data':[]}
with open("/home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/unlearned_test.json", 'r') as load_f:
    load_dict = json.load(load_f)
    print(len(load_dict['data']))
    for exa in load_dict['data']:
        cur_dict={"question_ents":[],"answers":[]}
        question = exa['question']
        entities = nlp(question)
        cur_dict["question"]=question
        cur_dict["id"]=exa['id']
        cur_dict["question_ents"]=list(set([word.text for word in entities.ents]))
        cur_dict['answers']=list(set(exa['answers']))
        # if len(cur_dict["question_ents"])>0 and len(cur_dict['answers'])>0:
        all_unlearned_dicts.append(cur_dict)
print(len(all_unlearned_dicts))


learned_dicts=[]
test_pair_num=len(all_unlearned_dicts)//2
with open("/home/dqx/neural_kb/cbqa/data_previous/tq_train.json", 'r') as load_f:
    cnt=0
    load_dict = json.load(load_f)

    for exa in load_dict['data']:
        if cnt==int(test_pair_num):
            break
        cur_dict={"question_ents":[],"answers":[]}
        question = exa['question']
        entities = nlp(question)
        cur_dict["question"]=question
        cur_dict["id"]=exa['id']
        cur_dict["question_ents"]=list(set([word.text for word in entities.ents]))
        cur_dict['answers']=list(set(exa['answers']))
        # if len(cur_dict["question_ents"])>0 and len(cur_dict['answers'])>0:
        all_unlearned_dicts.append(cur_dict)
        cnt+=1



random.shuffle(all_unlearned_dicts)
random.shuffle(learned_dicts)

learned_entpair_dict['data']=learned_dicts
unlearned_entpair_dict['data']=all_unlearned_dicts

with open("/home/dqx/neural_kb/cbqa/sup_meta/result/no_ffn_wq_128_t5-base_1e-3_tq/unlearned_all_entpairs.json", 'w') as write_f:
	json.dump(unlearned_entpair_dict, write_f, indent=4, ensure_ascii=False)
print(len(unlearned_entpair_dict['data']))
    
    # print(sent)
    # cur = 0
    # for word in entities.ents:
    #     #or change here
    #     # print(word)
        
    #     if word.label_ in ['PERSON', 'EVENT', 'FAC','LOC','WORK_OF_ART','GPE','NORP','LANGUAGE','LAW']:
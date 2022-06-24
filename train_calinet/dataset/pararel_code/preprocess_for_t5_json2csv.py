import json
import random
import os
import csv
with open("/home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_trex_500each.json",'r') as load_f:
    load_dict = json.load(load_f)
data=load_dict['data']
fact_num=75
data_subset = random.sample(data, fact_num)

output_path = "probing_data_" + str(fact_num)+"test/"
Details = ['src_sent', 'tgt_sent']
os.mkdir(output_path)
with open(output_path+"test.csv","w",encoding="utf-8") as csvfile_s:
    with open(output_path+"train.csv","w",encoding="utf-8") as csvfile_t:
        with open(output_path+"val.csv","w",encoding="utf-8") as csvfile_v:
            with open(output_path+"false_test.csv","w",encoding="utf-8") as csvfile_f:
                writer_t = csv.writer(csvfile_t)
                writer_t.writerow(Details)
                writer_v = csv.writer(csvfile_v)
                writer_v.writerow(Details)
                writer_s = csv.writer(csvfile_s)
                writer_s.writerow(Details)
                writer_f = csv.writer(csvfile_f)
                writer_f.writerow(Details)
                for d in data_subset:
                    entity_lst = []
                    entity_test_lst = []
                    entity_val_lst = []
                    entity_false_lst = []
                    sentences = d['sentences']
                    for sentence_id in range(len(sentences)):
                        if sentence_id%10 == 0:
                            entity_test_lst.append([d['sentences'][sentence_id][0], d['sentences'][sentence_id][1]])
                            entity_false_lst.append([d['sentences'][sentence_id][0], d['sentences'][sentence_id][2]])
                        elif sentence_id%10 == 1:
                            entity_val_lst.append([d['sentences'][sentence_id][0], d['sentences'][sentence_id][1]])
                        else:
                            entity_lst.append([d['sentences'][sentence_id][0], d['sentences'][sentence_id][1]])

                    entity_lst=list(set([tuple(t) for t in entity_lst]))
                    entity_test_lst=list(set([tuple(t) for t in entity_test_lst]))
                    entity_val_lst=list(set([tuple(t) for t in entity_val_lst]))
                    entity_false_lst=list(set([tuple(t) for t in entity_false_lst]))
                    writer_t.writerows(entity_lst)
                    writer_s.writerows(entity_test_lst)
                    writer_v.writerows(entity_val_lst)
                    writer_f.writerows(entity_false_lst)

        
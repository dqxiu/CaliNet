import json
import csv

with open("/home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_1000/t5_large_cp_score.json",'r') as load_f:
    load_dict = json.load(load_f)
    data_list = load_dict["data"]
relation_dict=dict()
cnt=0
with open('/home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_1000/t5_large_cp_score.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(["id","relation","score","fact","answer"])
    for item in data_list:
        # if int(item["CKA"])>1:
        # if int((item["P_true"]+0.0001)/(item["P_false"]+0.0001))>1 and int((item["P_true"]+0.001)/(item["P_false"]+0.001))<=1:
        # if item["relation"] not in relation_dict.keys():
            # relation_dict[item["relation"]]=dict()
            # relation_dict[item["relation"]]["known"]=[]
            # relation_dict[item["relation"]]["unknown"]=[]
            # relation_dict[item["relation"]]["d_known"]=[]
            # relation_dict[item["relation"]]["d_unknown"]=[]
        score=(item["P_true"]+0.001)/(item["P_false"]+0.001)
        item_score=[item["id"],item["relation"],score,item["src_true"],item["target_token"]]
        f_csv.writerow(item_score)
        # if (item["P_true"]+0.001)/(item["P_false"]+0.001)>1:
        #     relation_dict[item["relation"]]["known"].append(item["src_true"])
        #     if (item["P_true"]+0.001)/(item["P_false"]+0.001)>10:
        #         relation_dict[item["relation"]]["d_known"].append(item["src_true"])
        # else:
        #     relation_dict[item["relation"]]["unknown"].append(item["src_true"])
        #     if (item["P_true"]+0.001)/(item["P_false"]+0.001)<=0.1:
        #         relation_dict[item["relation"]]["d_unknown"].append(item["src_true"])

            

    # for key in relation_dict.keys():
    #     print(key)
    #     print(len(relation_dict[key]["known"]))
    #     print(len(relation_dict[key]["d_known"]))
    #     print(len(relation_dict[key]["unknown"]))
    #     print(len(relation_dict[key]["d_unknown"]))
    #     print("--------")

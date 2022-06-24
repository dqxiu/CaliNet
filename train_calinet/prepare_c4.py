from datasets import load_dataset
import random
import csv
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def racha_detection(lista):
    # It returns a list of lists where each sub-list contains the consecutive tokens in the list
    rachas = []
    racha = []
    for i, element in enumerate(lista):
        if (i<len(lista)-1) and (lista[i+1] == element+1):
            racha.append(element)
        else:
            if len(racha)>0:
                rachas.append(racha + [element])          
            else:# (i!=len(lista)-1):
                rachas.append([element])
            racha = []
    return rachas
def masking(tokenized_sentence, rachas):
    # Function to mask a tokenized_sentence (token ids) following the rachas described in rachas
    # Only one sentinel_token per racha
    sent_token_id = 0
    enmascared = tokenized_sentence.copy()
    for racha in rachas:
        sent_token = f'<extra_id_{sent_token_id}>'
        sent_id = tokenizer.encode(sent_token)[0]
        for i, idx in enumerate(racha):
            if i==0:
                enmascared[idx] = sent_id
            else:
                enmascared[idx] = -100
        sent_token_id += 1
        
    enmascared = [t for t in enmascared if t!=-100] 

    return enmascared

def add_noise(sentence, tokenizer, percent=0.05):
    # Function that takes a sentence, tokenizer and a noise percentage and returns
    # the masked input_ids and masked target_ids accordling with the T5 paper and HuggingFace docs
    # To see the process working uncomment all the prints ;)
    # tokenized_sentence = tokenizer.encode(sentence)
    tokenized_sentence = sentence.split(" ")
    #print('PRE-MASKED:')
    #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(tokenized_sentence)))
   
    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)), 
                                       int(len(tokenized_sentence)*percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas)
    #print('RACHAS INPUT: {}'.format(rachas))
    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas)
    #print('RACHAS TARGET: {}'.format(rachas))
    
    print('POST-MASKED:')
    output_src=[]
    output_tgt=[]
    for tok in enmascared_input:
        if not isinstance(tok,str):
            output_src.append(tokenizer.convert_ids_to_tokens(tok))
        else:
            output_src.append(tok)
    for tok in enmascared_target:
        if not isinstance(tok,str):
            output_tgt.append(tokenizer.convert_ids_to_tokens(tok))
        else:
            output_tgt.append(tok)
    # enmascared_input='{}'.format(" ".join(tokenizer.convert_ids_to_tokens(enmascared_input))).replace("▁"," ").replace("  "," ").replace("  "," ")
    # enmascared_target='{}'.format(" ".join(tokenizer.convert_ids_to_tokens(enmascared_target))).replace("▁"," ").replace("  "," ").replace("  "," ")
    enmascared_input=" ".join(output_src)
    enmascared_target=" ".join(output_tgt)
    return enmascared_input, enmascared_target
dataset = load_dataset("stas/c4-en-10k")
sentences=[]
for paragraph in dataset['train']['text'][0:1000]:
    sentences.extend(paragraph.split("\n"))
random.shuffle(sentences)
cnt=0
Details = ['src_sent', 'tgt_sent']
entity_lst=[]
with open("/home/dqx/neural_kb/fact_checker/dataset/pararel/probing_data_1000/c4_test.csv","w",encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(Details)
    for sent in sentences:
        entity_lst=[]
        if cnt>999:
            break
        print(sent)
        token_list=sent.split(" ")
        if len(token_list)>128 or len(token_list)<12:
            continue
        # replace_id = random.randint(0,len(token_list)-4)
        # replace_span = random.randint(1,4)
        # ans = token_list[replace_id:replace_id+replace_span]
        # src_sent = " ".join(token_list[0:replace_id])+" <extra_id_0> "+" ".join(token_list[replace_id+replace_span:])
        # tgt_sent = "<extra_id_0> " + " ".join(ans) + " <extra_id_1>"
        # output_tgt=tgt_sent
        # print(src_sent)
        # print(tgt_sent)
        # input()
        output_src,output_tgt=add_noise(sent,tokenizer)
        if "<extra_id_1>" not in output_tgt or "<extra_id_2>" in output_tgt:
            continue
        # entity_lst.append([src_sent, tgt_sent])
        cnt+=1
        
        entity_lst.append([output_src,output_tgt])
        writer.writerows(entity_lst)





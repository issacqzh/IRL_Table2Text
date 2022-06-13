# this file returns extractor models

from transformers.tokenization_utils_base import BatchEncoding
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, BertTokenizerFast, BertForQuestionAnswering, \
    DistilBertTokenizerFast, DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import re
import numpy as np
import json
from loader import *
import os
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import difflib
import sys
import unidecode

field2question = {
    "name" : "What is the name of the person?",
    "year_of_birth" : "When was the person born?",
    "place_of_birth" : "Where was the person born?",
    "place_of_death" : "Where did the person die?",
    "country" : "What is the country of citizenship?"
}

modeltype2tokenizerclass = {
    "bert" : (BertTokenizerFast, 'bert-base-uncased'),
    "roberta": (RobertaTokenizer, 'roberta-base'),
    "distil-bert-fast": (DistilBertTokenizerFast, 'distilbert-base-uncased'),
    "distil-bert": (DistilBertTokenizer, 'distilbert-base-uncased')
}

modeltype2modelclass = {
    "bert" : (BertForQuestionAnswering, 'bert-base-uncased'),
    "roberta": (RobertaForQuestionAnswering, 'roberta-base'),
    "distil-bert": (DistilBertForQuestionAnswering, 'distilbert-base-uncased'),
    "distil-bert-fast": (DistilBertForQuestionAnswering, 'distilbert-base-uncased')
}

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', return_dict=True)


def infer_extractor_model_hf_pretrained(field, context):
    tokenizer = AutoTokenizer.from_pretrained("twmkn9/bert-base-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/bert-base-uncased-squad2", return_dict=True)
    device = torch.device("cuda")
    model = model.to(device)
    model = model.eval()
    question = field2question[field]
    inputs = tokenizer(question, context, return_tensors='pt')
    inputs = inputs.to(device)

    outputs = model(**inputs)

    start_pos = torch.argmax(outputs.start_logits)
    end_pos = torch.argmax(outputs.end_logits)
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_pos:end_pos]))
    return answer



def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

def benchmark_off_the_shelf(field):
    tokenizer = AutoTokenizer.from_pretrained("twmkn9/bert-base-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/bert-base-uncased-squad2", return_dict=True)
    device = torch.device("cuda")
    model = model.to(device)
    model = model.eval()
    # question = field2question[field]

    train_data_filename = "extractor_train_data_"+field+".json"
    train_data = json.load(open(train_data_filename, "r"))
    total = len(train_data)
    count = 0
    batch_size = 16
    total_batches = int(total/batch_size)
    for i in trange(total_batches):
        context = [train_data[j][0] for j in range(i * batch_size , (i+1) * batch_size)]
        question = [train_data[j][1] for j in range(i * batch_size , (i+1) * batch_size)]
        gt_ans = [train_data[j][2].lower() for j in range(i * batch_size , (i+1) * batch_size)]
        
        # pred_ans = infer_extractor_model_hf_pretrained(field, context)
        
        inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(device)

        outputs = model(**inputs)

        start_pos = torch.argmax(outputs.start_logits, -1)
        end_pos = torch.argmax(outputs.end_logits, -1)
        # answer = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][j][start_pos[j]:end_pos[j]])) for j in range(batch_size)]
        for k in range(batch_size):
            gt = gt_ans[k]
            pred = answer[k]
            xyz = get_overlap(gt, pred)
            
            print("Ground truth = ", gt)
            print("Predicted full = ", pred)
            print("Predicted    = ", xyz)
            print("acc = ", float(len(xyz))/float(len(gt)) )
            print()
            # if gt_ans[k] == answer[k]:
                # count += 1
            count += float(len(xyz))/float(len(gt))

    print("acc = ", float(count)/float(total))

def get_extractor_models(field, model_type, load_saved_model):
    if field == "year_of_birth":
        raise NotImplementedError
    (model_class, pretrained_filename) = modeltype2modelclass[model_type]
    model = model_class.from_pretrained(pretrained_filename, return_dict=True)
    
    # load saved models
    if load_saved_model:
        pass

    return model

def prepare_batch(tokenizer, contexts, questions, start_pos_list, end_pos_list):
    appended_context_list = []
    target_token_start_list = []
    target_token_end_list = []
    
    for i in range(len(contexts)):
        appended_context =  questions[i] + " [SEP] " + contexts[i]
        appended_context_list.append(appended_context)

    inputs = tokenizer(appended_context_list, return_tensors='pt', padding=True)
    for i in range(len(contexts)):
        ans_start_char_id = int(start_pos_list[i]) + len(questions[i]) + len(" [SEP] ")
        ans_end_char_id = int(end_pos_list[i]) + len(questions[i]) + len(" [SEP] ")
        # print("answer = ", appended_context_list[i][ans_start_char_id : ans_end_char_id], " start char = ", appended_context_list[i][ans_start_char_id], " end char = ", appended_context_list[i][ans_end_char_id])
        
        
        ans_start_token_id = inputs.char_to_token(i,ans_start_char_id)
        ans_end_token_id = inputs.char_to_token(i,ans_end_char_id)
        target_token_start_list.append(int(ans_start_token_id))
        target_token_end_list.append(int(ans_end_token_id))
    
    target_token_start_list = torch.from_numpy(np.array(target_token_start_list)).long()
    target_token_end_list = torch.from_numpy(np.array(target_token_end_list)).long()
    
    return inputs, target_token_start_list, target_token_end_list


def prepare_cache_training_data(model_type, field):
    (tokenizer_class, pretrained_filename) = modeltype2tokenizerclass[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/")
    
    train_data_filename = "extractor_train_data_"+field+".json"
    cached_train_data_filename = "extractor_train_data_cached_"+field+".json"
    train_data = json.load(open(train_data_filename, "r"))
    total = len(train_data)
    appended_context_list = []
    target_token_start_list = []
    target_token_end_list = []
    
    cached_train_data = []
    for i in trange(len(train_data)):
        d =  train_data[i]
        # sample --> [context, question, answer, start_pos, end_pos]
        context = d[0]
        question = d[1]
        ans_start_char_id = int(d[3])
        ans_end_char_id = int(d[4])

        appended_context =  [question + " [SEP] " + context]
    
        inputs = tokenizer(appended_context, return_tensors='pt', padding=True, truncation=True)
    
        ans_start_char_id = ans_start_char_id + len(question) + len(" [SEP] ")
        ans_end_char_id = ans_end_char_id + len(question) + len(" [SEP] ")
        
        ans_start_token_id = inputs.char_to_token(0,ans_start_char_id)
        ans_end_token_id = inputs.char_to_token(0,ans_end_char_id)
        
        temp = [appended_context[0], ans_start_token_id, ans_end_token_id]
        cached_train_data.append(temp)
    
    json.dump(cached_train_data, open(cached_train_data_filename, "w"))
    


def train_extractor_model(model_type, field):
    (tokenizer_class, pretrained_filename) = modeltype2tokenizerclass[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/")
    (model_class, pretrained_filename) = modeltype2modelclass[model_type]
    model = model_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/", return_dict=True)
    device = torch.device("cuda")
    model = model.to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_data_filename = "extractor_train_data_cached_"+field+".json"
    train_data = json.load(open(train_data_filename, "r"))
    total = len(train_data)
    batch_size = 32
    num_epochs = 20
    num_batches = int(total/batch_size)
    for epoch in trange(num_epochs):
        start = 0
        ep_loss = 0
        lcount = 0
        for b_idx in trange(num_batches):
            
            # sample --> [appended_context, ans_start_token_id, ans_end_token_id]
            batch_train_data = train_data[b_idx * batch_size :(b_idx+1)*batch_size]
            contexts = [x[0] for x in batch_train_data]
            target_start = torch.from_numpy(np.array([x[1] for x in batch_train_data])).long()
            target_end = torch.from_numpy(np.array([x[2] for x in batch_train_data])).long()
            # print(type(contexts))
            # print(type(contexts[0]))
            inputs = tokenizer(contexts, return_tensors='pt', padding=True, truncation=True)
            
            inputs = inputs.to(device)
            target_start = target_start.to(device)
            target_end = target_end.to(device)
            

            
            
            outputs = model(**inputs, start_positions=target_start, end_positions=target_end)
            
            loss = outputs.loss
            loss = loss.mean()
            ep_loss += loss.item()
            lcount += 1
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        print("Epoch avg. loss = ", float(ep_loss) / float(lcount) )
        # save model
        folder_path = "./finetuned_qa_"+field+"/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        model_name = "model_ep_"+str(epoch)+".pth"
        opt_name = "opt_ep_"+str(epoch)+".pth"
        if epoch % 5 == 0:
            torch.save(model.state_dict(), folder_path + model_name)
            torch.save(optimizer.state_dict(), folder_path + opt_name)
    
def get_extractor_tokenizer(model_type):
    (tokenizer_class, pretrained_filename) = modeltype2tokenizerclass[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/")
    return tokenizer

def get_all_extractor_models(model_type):
    all_models = {}
    device = torch.device("cuda")
    
    (model_class, pretrained_filename) = modeltype2modelclass[model_type]
    
    model1 = model_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/", return_dict=True)
    model_path = "./finetuned_qa_name/model_ep_15.pth"
    model1.load_state_dict(torch.load(model_path))
    model1 = model1.to(device)
    model1 = model1.eval()
    all_models["name"] = model1
    
    model2 = model_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/", return_dict=True)
    model_path ="./finetuned_qa_place_of_birth/model_ep_5.pth"
    model2.load_state_dict(torch.load(model_path))
    model2 = model2.to(device)
    model2 = model2.eval()
    all_models["place_of_birth"] = model2
    
    model3 = model_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/", return_dict=True)
    model_path = "./finetuned_qa_place_of_death/model_ep_5.pth"
    model3.load_state_dict(torch.load(model_path))
    model3 = model3.to(device)
    model3 = model3.eval()
    all_models["place_of_death"] = model3
    
    model4 = model_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/", return_dict=True)
    model_path = "./finetuned_qa_country/model_ep_15.pth"
    model4.load_state_dict(torch.load(model_path))
    model4 = model4.to(device)
    model4 = model4.eval()
    all_models["country"] = model4
    
    return all_models
    
def test_extractor_model(model_type, field, validation, model_path):
    (tokenizer_class, pretrained_filename) = modeltype2tokenizerclass[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/")
    (model_class, pretrained_filename) = modeltype2modelclass[model_type]
    model = model_class.from_pretrained(pretrained_filename, cache_dir="./hf_cache/", return_dict=True)
    device = torch.device("cuda")
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)

    
    model = model.eval()
    # question = field2question[field]
    if validation:
        test_data_filename = "extractor_val_data_"+field+".json"
    else:
        test_data_filename = "extractor_test_data_"+field+".json"

    test_data = json.load(open(test_data_filename, "r"))
        
    total = len(test_data)
    count = 0
    batch_size = 8
    total_batches = int(total/batch_size)
    for i in trange(total_batches):
        context = [test_data[j][0] for j in range(i * batch_size , (i+1) * batch_size)]
        question = [test_data[j][1] for j in range(i * batch_size , (i+1) * batch_size)]
        gt_ans = [test_data[j][2].lower() for j in range(i * batch_size , (i+1) * batch_size)]
        
        # pred_ans = infer_extractor_model_hf_pretrained(field, context)
        
        inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
        inputs = inputs.to(device)

        outputs = model(**inputs)

        start_pos = torch.argmax(outputs.start_logits, -1)
        end_pos = torch.argmax(outputs.end_logits, -1)
        answer = [tokenizer.decode(inputs["input_ids"][j][start_pos[j]:end_pos[j]].cpu().detach().numpy().tolist() ) for j in range(batch_size)]
        answer2 = [tokenizer.decode(inputs["input_ids"][j][start_pos[j]].cpu().detach().numpy().tolist() ) for j in range(batch_size)]
        
        
        for k in range(batch_size):
            gt = unidecode.unidecode(gt_ans[k])
            pred = answer[k]
            if pred == "":
                pred = "".join(answer2[k].split())
            if "," in gt:
                ind1 = gt.find(',')
                gt = gt[:ind1]
                gt = gt.strip()
            if "(" in gt:
                ind1 = gt.find('(')
                gt = gt[:ind1]
                gt = gt.strip()

            
            if "," in pred:
                ind1 = pred.find(',')
                pred = pred[:ind1]
                pred = pred.strip()
            if "(" in gt:
                ind1 = pred.find('(')
                pred = pred[:ind1]
                pred = pred.strip()
            xyz = get_overlap(gt, pred)
            print("Ground truth = ", gt)
            print("Predicted full = ", pred)
            print("Predicted    = ", xyz)
            print("acc = ", float(len(xyz))/float(len(gt)) )
            # print()
            # if gt_ans[k] == answer[k]:
                # count += 1
            count += float(len(xyz))/float(len(gt))

    print("acc = ", float(count)/float(total))


def infer_extractor_model_with_score(model, tokenizer, field, context, ground_truth):
    # print("In inference of extractor")
    # print("context = ", context)
    # print("total = ", len(context))
    device=torch.device("cuda")
    question = []
    for _ in range(len(context)):
        question.append(field2question[field])
        
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
    inputs = inputs.to(device)

    outputs = model(**inputs)
    
    start_pos = torch.argmax(outputs.start_logits, -1)
    end_pos = torch.argmax(outputs.end_logits, -1)
    # print("start pos = ", start_pos.size())
    # print("end pos = ", end_pos.size())
    scores = []
    for i in range(len(context)):
        # print("context =", context[i])
        answer = tokenizer.decode(inputs["input_ids"][i][start_pos[i]:end_pos[i]].cpu().detach().numpy().tolist() )
        answer2 = tokenizer.decode(inputs["input_ids"][i][start_pos[i]].cpu().detach().numpy().tolist() )

        answer2 = "".join(answer2.split())
        answer = answer.lower()
        answer2 = answer2.lower()
        # print("gt = ", ground_truth[i])
        # print("answer = ", answer)
        gt = ground_truth[i].lower()

        gt = unidecode.unidecode(gt)
        pred = answer
        if pred == "":
            pred = answer2
        if "," in gt:
            ind1 = gt.find(',')
            gt = gt[:ind1]
            gt = gt.strip()
        if "(" in gt:
            ind1 = gt.find('(')
            gt = gt[:ind1]
            gt = gt.strip()

        
        if "," in pred:
            ind1 = pred.find(',')
            pred = pred[:ind1]
            pred = pred.strip()
        if "(" in gt:
            ind1 = pred.find('(')
            pred = pred[:ind1]
            pred = pred.strip()
        xyz = get_overlap(gt, pred)
        score = float(len(xyz))/float(len(gt)) if len(gt) > 0 else 0
        scores.append(score)
        # sys.exit()
    return scores

def infer_extractor_model(model_path, model_type, field, context):
    question = field2question[field]
    (tokenizer_class, pretrained_filename) = modeltype2tokenizerclass[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_filename)
    (model_class, pretrained_filename) = modeltype2modelclass[model_type]
    model = model_class.from_pretrained(pretrained_filename, return_dict=True)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda")
    model = model.to(device)
    model = model.eval()
    
    # inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
    inputs = tokenizer(question, context, return_tensors='pt')
    inputs = inputs.to(device)
    print(inputs["input_ids"][0])
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    l = inputs["input_ids"][0].detach().cpu().numpy().tolist()
    print(l)
    print(type(l))
    string = tokenizer.decode(l)
    total = len(string)
    print(tokens)
    print(string)
    sep_index = (inputs["input_ids"][0] == tokenizer.sep_token_id).nonzero()
    # num_seg_a = sep_index + 1
    # num_seg_b = len(inputs["input_ids"][0]) - num_seg_a
    # segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    ans_start_char_id = 1  # 1- indexed
    ans_end_char_id = 11 # 1 - indexed

    ans_start_token_id = inputs.char_to_token(0,ans_start_char_id)
    word1 = inputs.char_to_word(0,ans_start_char_id)
    print("word1 = ",word1)
    print(ans_start_token_id)
    ans_end_token_id = inputs.char_to_token(0,ans_end_char_id)
    print(ans_end_token_id)
    print("START")
    for i in range(ans_start_char_id, total):
        word = inputs.char_to_word(0,i)
        tpos = inputs.char_to_token(0,i)
        
        print("i = ", i , " word = ", word, " token pos = ", tpos, " token = ",tokens[tpos])
        # print(i, inputs.char_to_token(0,i), tokenizer.decode([inputs.char_to_token(0,i)]))
    print("END")

    # dummy_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_pos:end_pos]))
    
    # dummy_answer = tokenizer.decode(inputs["input_ids"][0][list(range(ans_start_token_id, ans_end_token_id))])
    print(type(ans_start_token_id))
    print(type(ans_end_token_id))
    x = list(range(ans_start_token_id, ans_end_token_id))
    print(type(x))
    print(x)
    print(type(x[0]))
    dummy_answer = tokenizer.decode(list(range(ans_start_token_id, ans_end_token_id)))
    print(dummy_answer)
    print(inputs.keys())
    print(type(inputs["input_ids"]))
    print(inputs["input_ids"].size())
    print(inputs["token_type_ids"].size())
    print(inputs["token_type_ids"])
    print(inputs["attention_mask"].size())
    print(tokens)
    print(len(tokens))
    outputs = model(**inputs)

    start_pos = torch.argmax(outputs.start_logits)
    end_pos = torch.argmax(outputs.end_logits)
    print(outputs.start_logits.size())
    print(outputs.end_logits.size())
    print(sep_index)
    print(sep_index.size())
    # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_pos:end_pos]))
    answer = tokenizer.decode(inputs["input_ids"][0][start_pos:end_pos])
    return answer



def get_year_of_birth(context):
    # print(context)
    matches = re.findall(r'.([1-9][0-9]{3})', context)
    # print(matches)
    matches = [int(m) for m in matches]
    return int(np.min(np.array(matches)))

def generate_training_data(field, mode):
    # mode = 0 for training, 1 for validation, 2 for test
    mode_type = {
        0:"train",
        1:"val",
        2:"test"
    }
    t_dataset = Table2text_seq(mode, type=0, USE_CUDA=True, batch_size=16)
    
    new_train_data = []
    rename_field_map = {
        "name" : "<Name_ID>",
        "year_of_birth" : "",
        "place_of_birth" : "<place of birth>",
        "place_of_death" : "<place of death>",
        "country" : "<country of citizenship>"
    }
    for d in t_dataset.data:
        context = d[1]
        context = " ".join(context)
        question = field2question[field]
        all_fields = d[2]
        all_values = d[0]
        field_found = False
        for i in range(len(all_fields)):
            if all_fields[i] == rename_field_map[field]:
                answer = all_values[i]
                field_found = True
                break
        
        if field_found:
            answer = answer.strip()
            start_pos = context.find(answer)
            end_pos = start_pos + len(answer) - 1
            train_sample = [context, question, answer, start_pos, end_pos]
            new_train_data.append(train_sample)

    filename = "extractor_"+mode_type[mode]+"_data_"+field+".json"
    print("Training samples = ", len(new_train_data))
    json.dump(new_train_data, open(filename,"w"))


def generate_stats():
    count_dict = {}
    t_dataset = Table2text_seq(0, type=0, USE_CUDA=True, batch_size=16)
    for d in t_dataset.data:
        all_fields = d[2]
        all_fields = list(set(all_fields))
        for field in all_fields:
            if field not in count_dict.keys():
                count_dict[field] = 0
            
            count_dict[field] += 1

    from collections import OrderedDict
    count_dict = OrderedDict(sorted(count_dict.items(), key=lambda kv: kv[1], reverse=True))
    print(count_dict)

if __name__ == "__main__":
    # print(get_year_of_birth("This is 2020 . last year was 2019"))
    # print(get_year_of_birth("This is 2020."))

    # generate train data
    generate_training_data("name",0)
    generate_training_data("place_of_birth",0)
    generate_training_data("place_of_death",0)
    generate_training_data("country",0)

    # generate val data
    generate_training_data("name", 1)
    generate_training_data("place_of_birth", 1)
    generate_training_data("place_of_death", 1)
    generate_training_data("country", 1)


    # generate_stats()

    prepare_cache_training_data("distil-bert-fast", "name")
    prepare_cache_training_data("distil-bert-fast", "place_of_birth")
    prepare_cache_training_data("distil-bert-fast", "place_of_death")
    prepare_cache_training_data("distil-bert-fast", "country")

    
    # training call
    train_extractor_model("distil-bert-fast", "name")
    train_extractor_model("distil-bert-fast", "place_of_birth")
    train_extractor_model("distil-bert-fast", "place_of_death")
    train_extractor_model("distil-bert-fast", "country")

    # print("Testing name...")
    # model_path = "./finetuned_qa_name/model_ep_15.pth"
    # test_extractor_model("distil-bert-fast", "name", True, model_path)

    # print("Testing place of birth...")
    # model_path = "./finetuned_qa_place_of_birth/model_ep_5.pth"
    # test_extractor_model("distil-bert", "place_of_birth", True, model_path)

    # print("Testing place of death...")
    # model_path = "./finetuned_qa_place_of_death/model_ep_15.pth"
    # test_extractor_model("distil-bert", "place_of_death", True, model_path)

    # print("Testing country...")
    # model_path = "./finetuned_qa_country/model_ep_15.pth"
    # test_extractor_model("distil-bert", "country", True, model_path)
    
    # inference call
    # model_path = "./finetuned_qa_name/model_ep_15.pth"
    # answer = infer_extractor_model(model_path, "bert", "name", "Jim Halpert is a popular guy in the office.")   
    # print(answer)

    # answer = infer_extractor_model_hf_pretrained("name", "Jim Halpert is a popular guy in the office.")
    # print(answer)

    # benchmark_off_the_shelf("name")
    # benchmark_off_the_shelf("place_of_birth")
    # benchmark_off_the_shelf("place_of_death")
    # benchmark_off_the_shelf("country")
    # pass

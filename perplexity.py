from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import tqdm
import torch
device = 'cuda'
model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id,cache_dir='/ssd-playpen/zheng/latest/cache',return_dict=True).to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_id,cache_dir='/ssd-playpen/zheng/latest/cache')
tokenizer.bos_token = '<s>'
def perplexity(partial_cand):
    max_length = model.config.n_positions
    stride = 256
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = '<s>'
    print(tokenizer.eos_token)
    print(tokenizer.unk_token)
    encodings = tokenizer(partial_cand,return_tensors='pt',padding=True,truncation=True,max_length=1024)
    print(encodings.input_ids)
    total_length = encodings.input_ids.clone().ne(50256).sum(1).to('cuda')
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to('cuda')
        print(input_ids)
        mask = input_ids.clone().ne(50256)
        
        target_ids = input_ids.clone().unsqueeze(2).long().to('cuda')
        # target_ids[:,:-stride] = -100
        with torch.no_grad():
                length = stride
                if end_loc >encodings.input_ids.size(1):
                        length = encodings.input_ids.size(1)-i
                outputs = model(input_ids,attention_mask=mask)
                print(outputs.loss)
                print(outputs.logits)
                logits = F.softmax(outputs.logits,dim=2)[:,:-1,:]
                output = logits.gather(2,target_ids[:,1:,:]).squeeze(2)[:,-length:]
                mask = mask[:,1:][:,-length:]
                output = output.log().mul(-1)*mask.float()
                lls.append(output.sum(1))
        return -torch.exp(torch.stack(lls).sum(0) / (total_length-1)).unsqueeze(1).cpu().numpy()


#partial_cand ='Béla Bodonyi ( born 14 December 1956 in Jászdózsa ) is a former Hungary Association football who played as a Forward (association football) . he played for the Hungary national football team .'
partial_cand=' '.join(['KG', '–', '1962', '–', '17 January 1903', 'in', 'London', 'that', 'was', 'the', 'seventh', 'child', 'nobleman', 'of', 'Sir', 'Sir James Hogg, 1st Baronet', 'London', 'sometimes', 'paid', 'conditions', '.', 'born', '14 February 1845', '–', '17 January 1903', 'Smith', 'was', 'an', 'eminent', 'English', 'amateur', 'military', 'athlete', 'who', 'played', 'for', 'Wanderers F.C.', 'in', 'the', 'Wanderers F.C.', 'now', 'called', 'Wanderers F.C.', 'Wanderers F.C.', 'as', 'now', 'part', 'of', 'Wanderers F.C.', 'suspended', 'now', 'now', 'called', 'Wanderers F.C.', 'now', 'Ryan', 'now', 'called', 'Sir James Hogg, 1st Baronet', 'of', 'Eton College', '.', 'john', 'Evans', 'was', 'born', 'to', 'Godwin', 'on', '14 February 1845', 'in', 'Scarborough', 'Street', 'London', 'England', "'s", 'exit', 'from', 'London', 'England', '.', 'in', '1582', 'he', 'graduated', 'from', 'Eton College', "'s", 'Eton College', 'the', 'Eton College', 'in', 'the', 'Wanderers F.C.', 'of', 'the', 'Wanderers F.C.', 'he', 'played', 'first-class'])
#partial_cand = [tokenizer.eos_token,'I am a doctor who was standing on the table']
print(perplexity(partial_cand))






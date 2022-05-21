import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re 
import numpy as np
import sys
import time
import torch
from eval_final import Evaluate
from eval import Evaluate_test
from extractor_models import *
import torch.nn.functional as F
from nltk.tag.stanford import StanfordNERTagger
from sner import Ner

class Rewards:
	def __init__(self,samples,gpt2_model,gpt2_tokenizer,eval_f,reward_names,all_extractor_models, extractor_tokenizer,extractor_field_mask_tensor):
		self.reward_names = reward_names
		self.tagger = Ner(host='localhost',port=9199)
		self.gpt2_tokenizer=gpt2_tokenizer
		self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
		self.gpt2_model=gpt2_model
		self.all_extractor_models = all_extractor_models
		self.extractor_tokenizer = extractor_tokenizer
		self.extractor_field_mask_tensor = extractor_field_mask_tensor
		self.eval = eval_f
		samples = list(zip(*samples))
		self.facts = samples[0]
		self.refs = samples[1]
		self.fields = samples[2]
		self.ref_entities=samples[6] # entities in the reference
		self.ref_orders = samples[7] # order of table facts in the reference
		self.years = samples[8] # strings, date of birth and death
		self.extractor_gold_answer = samples[9]

	def getRewards(self,full_cands):

		rewards = []
		all_refs = []
		
		for i in range(len(self.facts)):
			partial_cand = full_cands[i]
			if len(partial_cand)>1:
				all_refs.append(' '.join(partial_cand))
			else:
				all_refs.append('Spain spain')
			facts = self.facts[i]
			fields = self.fields[i]
			ref = self.refs[i]
			ref_entity = self.ref_entities[i]
			ref_order = self.ref_orders[i]
			years = self.years[i]

			r=[]
			for name in self.reward_names:
				if name == 'recall':
					r.append(self.recall(partial_cand,facts))
				elif name == 'repetition':
					r.append(self.repetition(partial_cand))
				elif name=='year_order':
					r.append(self.year_order(partial_cand,years))
				elif name =='fact_order':
					r.append(self.fact_order(partial_cand,ref_order,facts,fields))
				elif name == 'length':
					r.append(self.length(partial_cand))
				elif name =='bleu_4':
					r.append(self.bleu_4(partial_cand,ref))
				elif name == 'bleu_1':
					r.append(self.bleu_1(partial_cand,ref))
				elif name == 'precision':
					r.append(self.precision(partial_cand,facts))
			rewards.append(r)

		if 'extractor' in self.reward_names:
			final_scores = []
			small_batch_size = 8
			for i in range(0,len(all_refs),small_batch_size):
				raw_scores = self.extract_score(all_refs[i: i+small_batch_size], self.extractor_gold_answer[i:i+small_batch_size])
				modified_scores = torch.from_numpy(raw_scores) * self.extractor_field_mask_tensor[i:i+small_batch_size]
				temp_mask_sum = self.extractor_field_mask_tensor[i:i+small_batch_size].sum(1).view(-1)

				modified_scores = modified_scores.sum(1).view(-1) / temp_mask_sum
				modified_scores = modified_scores.view(-1,1)

				modified_scores = modified_scores.detach().cpu().numpy()
				final_scores.append(modified_scores)
			final_scores = np.concatenate(final_scores, 0)
			rewards = np.array(rewards)
			rewards = np.append(rewards,final_scores,1)
			
	
		if 'perplexity' in self.reward_names:
			stride = 1
			final_perplexity = []
			for i in range(0,len(all_refs),stride):
				final_perplexity.append(self.perplexity(all_refs[i:i+stride]))
			final_perplexity = np.concatenate(final_perplexity,0)
			rewards = np.append(rewards,final_perplexity,1)
		return rewards

	def extract_score(self,all_ref, gold_answers, score_return_type="AM"):
		extract_types = ['name', 'place_of_death', 'place_of_birth', 'country']
		scores = np.zeros((len(all_ref), len(extract_types)))
		
		for et_id in range(len(extract_types)):
			extract_type = extract_types[et_id]
			gold_answer = [gold_answers[i][extract_type] for i in range(len(gold_answers))]
			temp_score = infer_extractor_model_with_score(
				self.all_extractor_models[extract_type],
				self.extractor_tokenizer,
				extract_type,
				context = all_ref,
				ground_truth = gold_answer)
			scores[:, et_id] = np.array(temp_score)
		return scores

	def precision(self,partial_cand,facts):
		elements=self.tagger.get_entities(' '.join(partial_cand))
		all_entities=[]
		cur_ele=''
		cur_tag=''
		for element in elements:
			if element[1] != cur_tag:
				if len(cur_ele)>0 and cur_tag!='O':
					all_entities.append(cur_ele)
				cur_ele=element[0]
				cur_tag=element[1]
			else:
				cur_ele=cur_ele+' '+element[0]
		if cur_ele not in all_entities and cur_tag != 'O':
			all_entities.append(cur_ele)
		count = 0
		for entity in all_entities:
			for fact in facts:
				if fact == entity:
					count+=1
					break
		return count/len(all_entities)
		
	def recall(self,partial_cand,facts):
		count = 0
		for fact in facts:
			for word in partial_cand:
				if word == fact:
					count+=1
					break

		return count/len(facts)

	def repetition(self,partial_cand):

		trigrams = list(ngrams(partial_cand,3))

		if not trigrams: 
			return 0
		else:
			trigrams_set = set(ngrams(partial_cand,3))
			return len(trigrams_set)/len(trigrams)

	def year_order(self,partial_cand,years):

		sentence = ' '.join(partial_cand)
		cand_numbers = re.findall(r'\d+', sentence) 
		cand_years=[]
		for num in cand_numbers:
			if len(num)==4 and int(num) > 1500 and int(num)<2050:
				cand_years.append(num)
		
		if len(cand_years) < 2:
			return 0
		count = 0
		total = 0
		previous = 0
		while previous < len(cand_years) and cand_years[previous] not in years:
			previous+=1
		cur = previous+1
		while cur <len(cand_years):
			while cur<len(cand_years) and cand_years[cur] not in years:
				cur+=1
			if cur>=len(cand_years):
				break
			if years.index(cand_years[cur]) > years.index(cand_years[previous]):
				count += 1
			elif years.index(cand_years[cur]) < years.index(cand_years[previous]):
				count -= 1
			total +=1
			previous=cur
			cur+=1
		if total == 0:
			return total
		return count/total


	def fact_order(self,partial_cand,ref_order,facts,fields):
		fact_orders = []
		for token in partial_cand:
			for i in range(len(facts)):
				if token == facts[i] and fields[i] not in fact_orders and fields[i] in ref_order:
					fact_orders.append(fields[i])

		if len(fact_orders) <= 1:
			return 0
		count = 0
		for i in range(1,len(fact_orders)):
			if ref_order.index(fact_orders[i]) < ref_order.index(fact_orders[i-1]):
				count -= 1
			else:
				count += 1
		return count/(len(fact_orders)-1)

	def length(self, partial_cand):
		return len(partial_cand)/100

	def bleu_4(self, partial_cand,ref):
		ref = [x for x in ref if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
		ref = {0:[' '.join(ref)]}
		cand = {0: ' '.join(partial_cand)}

		final_scores = self.eval.evaluate(live=True,cand=cand, ref=ref)
		return final_scores['Bleu_4']*0.01

	def bleu_1(self, partial_cand,ref):
		ref = [x for x in ref if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
		ref = {0:[' '.join(ref)]}
		cand = {0: ' '.join(partial_cand)}

		final_scores = self.eval.evaluate(live=True,cand=cand, ref=ref)
		return final_scores['Bleu_1']*0.01

	def perplexity(self,partial_cand):
		max_length = self.gpt2_model.config.n_positions
		stride = 256
		encodings = self.gpt2_tokenizer(partial_cand,return_tensors='pt',padding=True,truncation=True,max_length=max_length)
		total_length = encodings.input_ids.clone().ne(50256).sum(1).to('cuda')
		lls = []
		for i in range(0, encodings.input_ids.size(1), stride):
			begin_loc = max(i + stride - max_length, 0)
			end_loc = i + stride
			input_ids = encodings.input_ids[:,begin_loc:end_loc].to('cuda')
			mask = input_ids.clone().ne(50256)
			target_ids = input_ids.clone().unsqueeze(2).long().to('cuda')
			with torch.no_grad():
				length = stride
				if end_loc >encodings.input_ids.size(1):
					length = encodings.input_ids.size(1)-i
				outputs = self.gpt2_model(input_ids,attention_mask=mask)
				logits = F.softmax(outputs.logits,dim=2)[:,:-1,:]
				output = logits.gather(2,target_ids[:,1:,:]).squeeze(2)[:,-length:]
				mask = mask[:,1:][:,-length:]
				output = output.log().mul(-1)*mask.float()
				lls.append(output.sum(1))
		return -torch.exp(torch.stack(lls).sum(0) / (total_length-1)).unsqueeze(1).cpu().numpy()




		





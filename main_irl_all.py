import argparse
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from random import sample
from predictor import Predictor
from loader import Table2text_seq
from structure.Encoder import EncoderRNN
from structure.Decoder import DecoderRNN
from structure.TopKDecoder import TopKDecoder
from structure.seq2seq import Seq2seq
from eval import Evaluate_test
from Rewards import Rewards
from Irl import Irl
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelWithLMHead
from extractor_models import get_all_extractor_models, get_extractor_tokenizer
from tqdm import trange


class Config(object):
    cell = "GRU"
    emsize = 256
    pemsize = 5
    nlayers = 1
    lr = 0.001
    s_epochs = 0
    irl_epochs = 20
    rl_epochs = 0
    batch_size = 64
    dropout = 0
    bidirectional = True
    max_grad_norm = 10
    max_len = 100
    irl_size = 500
    irl_lr = 0.1


class ConfigTest(object):
    cell = "GRU"
    emsize = 30
    pemsize = 30
    nlayers = 1
    lr = 0.001
    epochs = 2
    batch_size = 10
    dropout = 0
    bidirectional = True
    max_grad_norm = 1
    testmode = True
    max_len = 50
    irl_size = 100


parser = argparse.ArgumentParser(description='model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--load', type=str,  default='sl_params_P.pth',
                    help='path to load the model')
parser.add_argument('--save', type=str,  default='irl_params_P.pth',
                    help='path to save the final model')
parser.add_argument('--mode', type=int,  default=0,
                    help='train(0)/predict_individual(1)/predict_file(2)/compute score(3) or keep train (4)')
parser.add_argument('--type', type=int,  default=0,
                    help='person(0)/animal(1)')
parser.add_argument('--mask', type=int,  default=0,
                    help='false(0)/true(1)')
args = parser.parse_args()

warnings.filterwarnings("ignore")
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

path = 'irl_all_models'
if not os.path.exists(path):
  os.makedirs(path)

device = torch.device("cuda" if args.cuda else "cpu")
config = Config()
# config = ConfigTest()
if args.mask == 1:
    filepost = "_m"
else:
    filepost = ""

if args.type == 1:
    args.save = 'params_A.pkl'
    filepost += "_A.txt"
else:
    filepost += "_P.txt"

t_dataset = Table2text_seq(
    0, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
v_dataset = Table2text_seq(
    1, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
t_eval_dataset = Table2text_seq(
    2, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
print("number of training examples: %d" % t_dataset.len)

# pre-trained word vectors
# if args.type == 0:
#     embedding_path='./vocab.pkl'
# else:
#     embedding_path='./vocab_D_bert_embedding.pkl'
# with open('./vocab_glove_embedding.pkl','rb') as fp:
#     vocab_embedding=pickle.load(fp)
# vocab_embedding=torch.FloatTensor(vocab_embedding)
# embedding=nn.Embedding(vocab_embedding.size(0),vocab_embedding.size(1),padding_idx=0)
# embedding.weight=nn.Parameter(vocab_embedding)

embedding = nn.Embedding(t_dataset.vocab.size, config.emsize, padding_idx=0)
encoder = EncoderRNN(t_dataset.vocab.size, embedding, config.emsize, t_dataset.max_p, config.pemsize,
                     input_dropout_p=config.dropout, dropout_p=config.dropout, n_layers=config.nlayers,
                     bidirectional=config.bidirectional, rnn_cell=config.cell, variable_lengths=True)
decoder = DecoderRNN(t_dataset.vocab, t_dataset.vocab.size, embedding, config.emsize, config.pemsize, sos_id=3, eos_id=2, unk_id=1,
                     n_layers=config.nlayers, rnn_cell=config.cell, bidirectional=config.bidirectional,
                     input_dropout_p=config.dropout, dropout_p=config.dropout, USE_CUDA=args.cuda, mask=args.mask)
model = Seq2seq(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
predictor = Predictor(model, v_dataset.vocab, args.cuda)
weights = [0]*5
reward_names = ['repetition', 'recall', 'bleu_4', 'extractor', 'perplexity']
normalization_rewards = ['perplexity']
thresholds = {}
irl = Irl(weights, config.irl_size, config.irl_lr)

gpt2_model = GPT2LMHeadModel.from_pretrained(
    'gpt2', return_dict=True).to('cuda')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
all_extractor_models=get_all_extractor_models("distil-bert-fast")
extractor_tokenizer=get_extractor_tokenizer("distil-bert-fast")

def train_rewards(ref_rewards, cand_rewards, cand_probs, irl, irl_size, seed, cuda, norm_max_min, change_thresholds):
    random.seed(seed)
    indice=random.sample(range(ref_rewards.shape[0]), irl_size)

    sample_ref_rewards=ref_rewards[indice]
    sample_cand_rewards=cand_rewards[indice]
    sample_cand_probs=cand_probs[indice]
    print('ref rewards: ', np.sum(sample_ref_rewards, 0))
    print('cand rewards: ', np.sum(sample_cand_rewards, 0))

    ratios={}
    sample_ref_rewards=np.transpose(sample_ref_rewards)
    sample_cand_rewards=np.transpose(sample_cand_rewards)
    # set threshold
    for r in thresholds.keys():
        r_index=reward_names.index(r)
        r_threshold=thresholds[r]
        ratios[r]=np.sum(sample_cand_rewards[r_index]) / \
                         np.sum(sample_ref_rewards[r_index])
        sample_ref_rewards[r_index]=sample_ref_rewards[r_index]*r_threshold


    for n in norm_max_min.values():
        # normalization index in rewards, max, min
        sample_ref_rewards[n[0]]=(sample_ref_rewards[n[0]]-n[2])/(n[1]-n[2])
        sample_cand_rewards[n[0]]=(sample_cand_rewards[n[0]]-n[2])/(n[1]-n[2])
    sample_ref_rewards=np.transpose(sample_ref_rewards)
    sample_cand_rewards=np.transpose(sample_cand_rewards)

    relax_weights=irl.update(
        sample_cand_rewards, sample_ref_rewards, sample_cand_probs)
    # update thresholds
    if change_thresholds:
        for r in thresholds.keys():
            r_index=reward_names.index(r)
            if relax_weights[r_index] == True:
                thresholds[r] += 0.1
            else:
                thresholds[r]=(ratios[r]+thresholds[r])/2
            if thresholds[r] > 1:
                thresholds[r]=1


    print('relax weights: ', relax_weights)
    print('ratios: ', ratios)
    print('multipliers: ', thresholds)
    print('ref rewards: ', np.sum(sample_ref_rewards, 0))
    print('cand rewards: ', np.sum(sample_cand_rewards, 0))
    print('after update: ', irl.normalized_weights)


def train_batch(dataset, batch_idx, model, teacher_forcing_ratio, train_mode, weights, batch_cand_rewards, gpt2_model, gpt2_tokenizer, all_extractor_models, extractor_tokenizer, eval_f, reward_names, norm_max_min=None):
    batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov, list_oovs, targets, samples, w2fs, extractor_field_mask_tensor=dataset.get_batch(
        batch_idx)
    reward_cal=Rewards(samples, gpt2_model, gpt2_tokenizer, eval_f, reward_names,
                       all_extractor_models, extractor_tokenizer, extractor_field_mask_tensor)

    if train_mode == 'sl':
        losses=model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb, source_len, batch_t,
            batch_o_t, teacher_forcing_ratio, train_mode, reward_cal, weights, list_oovs, w2fs, batch_cand_rewards=batch_cand_rewards, norm_max_min=norm_max_min)
    else:  # rl
        losses, rl_probs, rl_rewards=model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb, source_len, batch_t,
            batch_o_t, teacher_forcing_ratio, train_mode, reward_cal, weights, list_oovs, w2fs, batch_cand_rewards=batch_cand_rewards, norm_max_min=norm_max_min)

    batch_loss=losses.mean()
    model.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    optimizer.step()
    if train_mode == 'sl':
        return batch_loss.item(), len(source_len)
    else:
        return batch_loss.item(), len(source_len), rl_probs, rl_rewards


def train_generator(t_dataset, v_dataset, model, n_epochs, teacher_forcing_ratio, weights, eval_f, train_mode, batch_size, t_cand_rewards=None, norm_max_min=None, gpt2_model=None, gpt2_tokenizer=None, reward_names=None, irl_epoch=None, all_extractor_models=None, extractor_tokenizer=None):
    best_dev=0
    epoch_score=0
    train_loader=t_dataset.corpus
    len_batch=len(train_loader)
    epoch_examples_total=t_dataset.len
    r_cand_rewards=[]
    r_cand_probs=[]

    for epoch in range(1, n_epochs+1):

        model.train(True)
        torch.set_grad_enabled(True)
        epoch_loss=0

        for batch_idx in range(len_batch):
            if train_mode == 'rl':
                batch_cand_rewards=t_cand_rewards[batch_idx * \
                    batch_size:(batch_idx+1)*batch_size]
                loss, num_examples, rl_probs, rl_rewards=train_batch(t_dataset, batch_idx, model, teacher_forcing_ratio, train_mode, weights,
                                                                     batch_cand_rewards, gpt2_model, gpt2_tokenizer, all_extractor_models, extractor_tokenizer, eval_f, reward_names, norm_max_min)
                if epoch == n_epochs:
                    r_cand_rewards.append(rl_rewards)
                    r_cand_probs.append(rl_probs)
            else:
                batch_cand_rewards=None
                loss, num_examples=train_batch(t_dataset, batch_idx, model, teacher_forcing_ratio, train_mode, weights, batch_cand_rewards,
                                               gpt2_model, gpt2_tokenizer, all_extractor_models, extractor_tokenizer, eval_f, reward_names, norm_max_min)
            epoch_loss += loss * num_examples
            sys.stdout.write(
                '%d batches processed. current batch loss: %f\r' %
                (batch_idx, loss)
            )
            sys.stdout.flush()


        epoch_loss /= epoch_examples_total
        log_msg="Finished epoch %d with losses: %.4f" % (epoch, epoch_loss)
        print(log_msg)

        if epoch % 5 == 0 or epoch == n_epochs:
            beam_search_model=Seq2seq(
                model.encoder, TopKDecoder(model.decoder, 3))
            beam_search_model.eval()
            predictor=Predictor(beam_search_model, v_dataset.vocab, args.cuda)
            print("Start Evaluating")
            cand, ref, ref_rewards, cand_rewards=predictor.preeval_batch(v_dataset, gpt2_model=gpt2_model, gpt2_tokenizer=gpt2_tokenizer,
                                                                         all_extractor_models=all_extractor_models, extractor_tokenizer=extractor_tokenizer, eval_f=eval_f, reward_names=reward_names)
            print('Result:')
            for i in range(13, 15):
                print('sample: ', i)
                print('ref: ', ref[i+1][0])
                print('cand: ', cand[i+1])
                print('cand_rewards', cand_rewards[0][i])

            print('ref rewards: ', np.sum(np.concatenate(ref_rewards), 0))
            print('cand rewards: ', np.sum(np.concatenate(cand_rewards), 0))
            final_scores=eval_f.evaluate(live=True, cand=cand, ref=ref)
            print('Bleu_1: ', np.mean(final_scores['Bleu_1']))
            print('Bleu_2: ', np.mean(final_scores['Bleu_2']))
            print('Bleu_3: ', np.mean(final_scores['Bleu_3']))
            print('Bleu_4: ', np.mean(final_scores['Bleu_4']))
            print('ROUGE_L: ', np.mean(final_scores['ROUGE_L']))
            epoch_score=np.mean(2*final_scores['ROUGE_L']*final_scores['Bleu_4'] / \
                (final_scores['Bleu_4'] + final_scores['ROUGE_L']))
#        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},'pretrain_3000_'+args.save)



        if epoch_score > best_dev:
            torch.save({'model_state_dict': model.state_dict(
            ), 'optimizer_state_dict': optimizer.state_dict()}, 'irl_all_models/'+str(irl_epoch)+'_'+args.save)
            print("model saved")
            best_dev=epoch_score
    if train_mode == 'sl':
        predictor=Predictor(model, t_eval_dataset.vocab, args.cuda)
        ref_rewards, t_cand_rewards, t_cand_probs=predictor.preeval_batch(t_eval_dataset, last_sl_epoch=1, gpt2_model=gpt2_model, gpt2_tokenizer=gpt2_tokenizer,
                                                                          all_extractor_models=all_extractor_models, extractor_tokenizer=extractor_tokenizer, eval_f=eval_f, reward_names=reward_names)
        return ref_rewards, t_cand_rewards, t_cand_probs
    else:  # rl
        return np.concatenate(r_cand_rewards), np.concatenate(r_cand_probs)


def train_process(t_dataset, v_dataset, model, s_epochs, irl_epochs, rl_epochs, irl, batch_size, gpt2_model, gpt2_tokenizer, all_extractor_models, extractor_tokenizer, reward_names, normalization_rewards):
    checkpoint=torch.load(args.load)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("model restored")
    eval_f=Evaluate_test()
    seeds=[]
    for i in range(irl_epochs):
        temp_seeds=[]
        for j in range(2):
           temp_seeds.append(random.random())
        seeds.append(temp_seeds)
    ref_rewards, t_cand_rewards, t_cand_probs=train_generator(t_dataset, v_dataset, model, s_epochs, 1, irl.normalized_weights, eval_f, 'sl', batch_size, gpt2_model=gpt2_model,
                                                              gpt2_tokenizer=gpt2_tokenizer, all_extractor_models=all_extractor_models, extractor_tokenizer=extractor_tokenizer, reward_names=reward_names)
    print('sl ends')
    # normalization
    norm_max_min={}
    ref_r=np.transpose(np.concatenate(ref_rewards))
    cand_r=np.transpose(np.concatenate(t_cand_rewards))

    if normalization_rewards:

        # max and min
        for n_r in normalization_rewards:
            n_r_index=reward_names.index(n_r)
            ref_n_r=ref_r[n_r_index]
            ref_max=np.max(ref_n_r)
            ref_min=np.min(ref_n_r)

            cand_n_r=cand_r[n_r_index]
            cand_max=np.max(cand_n_r)
            cand_min=np.min(cand_n_r)
            n_r_max=np.max([ref_max, cand_max])
            n_r_min=np.min([ref_min, cand_min])
            # index of normalization reward, max, min
            norm_max_min[n_r]=[n_r_index, n_r_max, n_r_min]


    ref_rewards=np.transpose(ref_r)
    t_cand_rewards=np.transpose(cand_r)
    t_cand_probs=np.concatenate(t_cand_probs)

    r_cand_rewards=None
    r_cand_probs=None
    for i in range(irl_epochs):
        for j in range(2):
            change_thresholds=True if j == 0 else False
            print('irl starts')
            if i == 0:
                train_rewards(ref_rewards, t_cand_rewards, t_cand_probs, irl,
                              irl.size, seeds[i][j], args.cuda, norm_max_min, change_thresholds)
            else:
                train_rewards(ref_rewards, r_cand_rewards, r_cand_probs, irl,
                              irl.size, seeds[i][j], args.cuda, norm_max_min, change_thresholds)
        print('rl starts')
        r_cand_rewards, r_cand_probs=train_generator(t_dataset, v_dataset, model, 5, 1, irl.normalized_weights, eval_f, 'rl', batch_size, t_cand_rewards, norm_max_min, gpt2_model=gpt2_model,
                                                     gpt2_tokenizer=gpt2_tokenizer, all_extractor_models=all_extractor_models, extractor_tokenizer=extractor_tokenizer, reward_names=reward_names, irl_epoch=i)
    if rl_epochs:
        r_cand_rewards, r_cand_probs=train_generator(t_dataset, v_dataset, model, rl_epochs, 1, irl.normalized_weights, eval_f, 'rl', batch_size, t_cand_rewards, norm_max_min,
                                                     gpt2_model=gpt2_model, gpt2_tokenizer=gpt2_tokenizer, all_extractor_models=all_extractor_models, extractor_tokenizer=extractor_tokenizer, reward_names=reward_names)



if __name__ == "__main__":
    if args.mode == 0:
        # train
        try:
            print("start training...")
            train_process(t_dataset, v_dataset, model, config.s_epochs, config.irl_epochs, config.rl_epochs, irl, config.batch_size,
                          gpt2_model, gpt2_tokenizer, all_extractor_models, extractor_tokenizer, reward_names, normalization_rewards)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

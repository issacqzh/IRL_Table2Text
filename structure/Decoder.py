import torch.nn as nn
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from .baseRNN import BaseRNN



class DecoderRNN(BaseRNN):

    def __init__(self,vocab, vocab_size, embedding, embed_size, pemsize, sos_id, eos_id,
                 unk_id, max_len=100, n_layers=1, rnn_cell='gru',
                 bidirectional=True, input_dropout_p=0, dropout_p=0,
                 lmbda=1.5, USE_CUDA = torch.cuda.is_available(), mask=0):
        
        hidden_size = embed_size
        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.vocab=vocab
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.mask = mask
        self.embedding = embedding
        self.lmbda = lmbda
        self.USE_CUDA = USE_CUDA
        #directions
        self.Wh = nn.Linear(hidden_size * 2, hidden_size)
        #output
        self.V = nn.Linear(hidden_size * 3, self.output_size)
        #params for attention
        self.Wih = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder input
        self.Wfh = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder field
        self.Ws = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.w_c = nn.Linear(1, hidden_size)  # for obtaining e from context vector
        self.v = nn.Linear(hidden_size, 1)
        # parameters for p_gen
        self.w_ih = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_fh = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_s = nn.Linear(hidden_size, 1)    # for changing hidden state into a scalar
        self.w_x = nn.Linear(embed_size, 1)     # for changing input embedding into a scalar
        # parameters for self attention
        self_size = pemsize * 2  # hidden_size +
        self.wp = nn.Linear(self_size, self_size)
        self.wc = nn.Linear(self_size, self_size)
        self.wa = nn.Linear(self_size, self_size)

    def get_matrix(self, encoderp):
        tp = torch.tanh(self.wp(encoderp))
        tc = torch.tanh(self.wc(encoderp))
        f = tp.bmm(self.wa(tc).transpose(1, 2))
        return F.softmax(f, dim=2)

    def self_attn(self, f_matrix, encoderi, encoderf):
        c_contexti = torch.bmm(f_matrix, encoderi)
        c_contextf = torch.bmm(f_matrix, encoderf)
        return c_contexti, c_contextf

    def rl_sample(self, combined_vocab,batch_size, list_oovs):
        # print("batch size = ", batch_size)
        # SOS 3
        # EOS 2
        # PAD 0
        rl_vocab=combined_vocab.clone().detach()

        rl_vocab[:,self.unk_id]= 0
        # rl_vocab[:, self.sos_id] = 0
        # rl_vocab[:, 0] = 0
        end_id_list = []
        for i in range(batch_size):
            end_id = self.vocab.size + len(list_oovs[i])
            end_id_list.append(end_id)

            rl_vocab[i][end_id:] = 0

        try:
            sample_idx = torch.multinomial(rl_vocab, 1).long()
        except:
            print(rl_vocab)
            print(rl_vocab.size())
            print(torch.isnan(rl_vocab).sum())
            print((rl_vocab<0).sum())
            print((rl_vocab==0).sum(0))
            print('ss')
            print((rl_vocab==0).sum(1))
            print((rl_vocab==0).sum())
            sys.exit()
        
        greedy_idx = combined_vocab.topk(1)[1]
        sample_idx=sample_idx.view(batch_size,1)
        greedy_idx=greedy_idx.view(batch_size,1)
        return sample_idx,greedy_idx


    def decode_step(self, input_ids, coverage, _h, enc_proj, batch_size, max_enc_len,
                    enc_mask, c_contexti, c_contextf, embed_input, max_source_oov, f_matrix):
        batch_size = input_ids.size(0)
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        e_t = self.v(torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))
        
        # mask to -INF before applying softmax
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.data.masked_fill_(enc_mask.data.byte(), -10000.0)
        attn_scores = F.softmax(attn_scores, dim=1)
        
        contexti = attn_scores.unsqueeze(1).bmm(c_contexti).squeeze(1)
        contextf = attn_scores.unsqueeze(1).bmm(c_contextf).squeeze(1)

        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, contexti, contextf), 1)), dim=1)
        
        # p_gen calculation
        p_gen = torch.sigmoid(self.w_ih(contexti) + self.w_fh(contextf) + self.w_s(_h) + self.w_x(embed_input))
        p_gen = p_gen.view(-1, 1)
        weighted_Pvocab = p_vocab * p_gen
        weighted_attn = (1-p_gen) * attn_scores
        
        if max_source_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros(batch_size, max_source_oov)
            if self.USE_CUDA:
                ext_vocab=ext_vocab.cuda()
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        del weighted_Pvocab       # 'Recheck OOV indexes!'
        
        # scatter article word probs to combined vocab prob.
        combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
        
        return combined_vocab, attn_scores

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_mask=None, encoder_hidden=None, encoderi=None, encoderf=None,
                encoderp=None, teacher_forcing_ratio=None,train_mode = None,reward_cal = None,list_oovs=None,weights=None,w2fs=None,fig=False,batch_cand_rewards = None,norm_max_min=None):
        
        # max_source_oov:length of source oov
        # input_ids: batch_s
        # targets: batch_t
        # target_id: batch_o_t
        
        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, encoder_hidden, encoderi, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_hidden_rl = self._init_state(encoder_hidden)
        
        coverage = torch.zeros(batch_size, max_enc_len)
        coverage_rl = torch.zeros(batch_size, max_enc_len)
        
        enci_proj = self.Wih(encoderi.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        enci_proj_rl = self.Wih(encoderi.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        
        encf_proj = self.Wfh(encoderf.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        encf_proj_rl = self.Wfh(encoderf.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        
        f_matrix = self.get_matrix(encoderp)
        f_matrix_rl = self.get_matrix(encoderp)
        
        enc_proj = enci_proj + encf_proj
        enc_proj_rl = enci_proj_rl + encf_proj_rl

        # get link attention scores -> S*, V*
        c_contexti, c_contextf = self.self_attn(f_matrix, encoderi, encoderf)
        c_contexti_rl, c_contextf_rl = self.self_attn(f_matrix_rl, encoderi, encoderf)
        
        
        if self.USE_CUDA:
            coverage = coverage.cuda()
            coverage_rl = coverage_rl.cuda()
        
        if teacher_forcing_ratio:
            
            embedded = self.embedding(targets)
            embed_inputs = self.input_dropout(embedded)
            # coverage initially zero
            dec_lens = (targets > 0).float().sum(1)
            
            lm_loss, cov_loss = [], []
            hidden, _ = self.rnn(embed_inputs, decoder_hidden)

            for _step in range(max_length):
                _h = hidden[:, _step, :]
                target_id = targets_id[:, _step+1].unsqueeze(1) 
                embed_input = embed_inputs[:, _step, :]

                combined_vocab, attn_scores = self.decode_step(input_ids, coverage, _h, enc_proj, batch_size,
                                                       max_enc_len, enc_mask, c_contexti, c_contextf, embed_input, max_source_oov, f_matrix)
                

                
                # mask the output to account for PAD
                target_mask_0 = target_id.ne(0).detach()
                
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                
                lm_loss.append(output.log().mul(-1) * target_mask_0.float())
                coverage = coverage + attn_scores

                # Coverage Loss
                # take minimum across both attn_scores and coverage
                _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
                cov_loss.append(_cov_loss.sum(1))
            
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda * torch.stack(cov_loss, 1).sum(1).div(dec_lens)
            
            if train_mode == 'rl':
                
                # print('here is the rl')
                rl_loss_steps=[]
                # greedy_lengths = np.array([max_length] * batch_size)
                max_length = self.max_length
                sample_lengths=np.array([max_length] * batch_size)
                sample_end_=torch.ones([batch_size,1]).detach()
                # greedy_end_ = torch.ones([batch_size,1]).detach()
                if self.USE_CUDA:
                    sample_end_ = sample_end_.cuda()
                    # greedy_end_ = greedy_end_.cuda()
                sample_words={}
                # greedy_words={}

                # sampled
                # targets_copy = targets.clone()
                targets_rl = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
                if self.USE_CUDA:
                    targets_rl = targets_rl.cuda()
                embed_input_rl = self.embedding(targets_rl)
                
                for _step in range(max_length):
                    
                    _h, _c = self.rnn(embed_input_rl, decoder_hidden_rl)
                    
                    combined_vocab, attn_scores = self.decode_step(input_ids, coverage_rl,
                                                           _h.squeeze(1), enc_proj_rl, batch_size, max_enc_len, enc_mask,
                                                           c_contexti_rl, c_contextf_rl, embed_input_rl.squeeze(1),
                                                           max_source_oov, f_matrix_rl)                    
                    
                    sample_idx,greedy_idx=self.rl_sample(combined_vocab,batch_size, list_oovs)
                    

                    # compute the log probability
                    rl_mask_0 = sample_idx.ne(0).detach() #PAD
                    rl_mask_2 = sample_idx.ne(2).detach() #EOS
                    rl_mask_3 = sample_idx.ne(3).detach() #SOS
                    sample_end_point = sample_idx.eq(2).detach() #EOS
                    sample_end_=sample_end_*rl_mask_2.float()
                    
                    if sample_end_point.dim() > 0:
                        
                        sample_end_point = sample_end_point.view(-1).cpu().numpy()
                        update_idx = ((sample_lengths > _step) & sample_end_point) != 0
                    
                    if _step == 0:
                        sample_lengths[update_idx] = 1
                    else:
                        sample_lengths[update_idx] = _step

                    
                    rl_output=combined_vocab.gather(1,sample_idx).add_(sys.float_info.epsilon)
                    # rl_loss_steps.append(rl_output.log().mul(-1)*rl_mask_0.float()*rl_mask_2.float()*rl_mask_3.float()*sample_end_)
                    rl_loss_steps.append(rl_output.log().mul(-1)*rl_mask_0.float()*rl_mask_3.float()*sample_end_)
                    
                    for i in range(batch_size):
                        if i not in sample_words:
                            sample_words[i]=[]
                        if sample_end_[i].item()!=0:
                            word = sample_idx[i].item()
                            if word <self.vocab.size:
                                word = self.vocab.idx2word[word]
                            else:
                                word = list_oovs[i][word-self.vocab.size]
                                
                            if word != '<PAD>' and word != '<EOS>' and word != '<SOS>':
                                sample_words[i].append(word)
                               
                    current_idx = sample_idx.clone().detach()
                    # change unk to corresponding field
                    
                    for i in range(batch_size):
                        w2f = w2fs[i]
                        if current_idx[i].item() > self.vocab_size-1:
                            current_idx[i] = w2f[current_idx[i].item()]

                    # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
                    embed_input_rl = self.embedding(current_idx)
                    decoder_hidden_rl = _c
                    coverage_rl = coverage_rl + attn_scores

                sample_rewards =reward_cal.getRewards(sample_words)
                raw_rewards = np.copy(sample_rewards)
                normalized_sample_rewards = np.transpose(sample_rewards)
                normalized_baseline_rewards = np.transpose(batch_cand_rewards)

                for n in norm_max_min.values():
                    # normalization index in rewards, max, min
                    normalized_sample_rewards[n[0]] = (normalized_sample_rewards[n[0]]-n[2])/(n[1]-n[2])
                    normalized_baseline_rewards[n[0]] = (normalized_baseline_rewards[n[0]]-n[2])/(n[1]-n[2])

                normalized_sample_rewards = np.transpose(normalized_sample_rewards)
                normalized_baseline_rewards = np.transpose(normalized_baseline_rewards)
                total_normalized_sample_rewards = list(np.dot(normalized_sample_rewards,weights))
                total_normalized_baseline_rewards = list(np.dot(normalized_baseline_rewards,weights))
                
                
                score_differences = [x[0]-x[1] for x in zip(total_normalized_sample_rewards,total_normalized_baseline_rewards)]
                
                score_differences=torch.FloatTensor(score_differences)
                update = (score_differences>0).detach()
                sample_lengths = torch.FloatTensor(sample_lengths.tolist())
                if self.USE_CUDA:
                    score_differences=score_differences.cuda()
                    sample_lengths = sample_lengths.cuda()
                    update = update.cuda()
                rl_cand_probs=torch.cat(rl_loss_steps, 1).sum(1).div(sample_lengths)
                rl_loss=rl_cand_probs.mul(score_differences).mul(update)

                if torch.isnan(rl_loss).sum()>0 or torch.isnan(total_masked_loss).sum()>0:
                    print('sample_length: ',sample_lengths)
                    print('sample_reward: ',raw_rewards)
                    print('samplew words: ',sample_words)
                    print('rl-loss steps: ',rl_loss_steps)

                
                return rl_loss*0.9+total_masked_loss*0.1, rl_cand_probs.clone().detach().cpu().view(-1).numpy(), raw_rewards

                #return total_masked_loss*0.03+rl_loss*0.97
            return total_masked_loss
        
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig)

    def evaluate(self, targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig):
        
        lengths = np.array([max_length] * batch_size)
        decoded_outputs = []
        if fig:
            attn = []
        embed_input = self.embedding(targets)
        probs = []
        end_=torch.ones([batch_size,1]).detach()
        if self.USE_CUDA:
            end_ = end_.cuda()
        # step through decoder hidden states
        for _step in range(max_length):
            _h, _c = self.rnn(embed_input, decoder_hidden)
            combined_vocab, attn_scores = self.decode_step(input_ids, coverage,
                                                           _h.squeeze(1), enc_proj, batch_size, max_enc_len, enc_mask,
                                                           c_contexti, c_contextf, embed_input.squeeze(1),
                                                           max_source_oov, f_matrix)
            # not allow decoder to output UNK
        
            combined_vocab[:, self.unk_id] = 0
            symbols = combined_vocab.topk(1)[1]
            
            if self.mask == 1:
                print('mask=1')
                tmp_mask = torch.zeros_like(enc_mask, dtype=torch.uint8)
                for i in range(symbols.size(0)):
                    pos = (input_ids[i] == symbols[i]).nonzero()
                    if pos.size(0) != 0:
                        tmp_mask[i][pos] = 1
                enc_mask = torch.where(enc_mask > tmp_mask, enc_mask, tmp_mask)

        
            if fig:
                
                attn.append(attn_scores)
            decoded_outputs.append(symbols.clone().detach())

            rl_mask_0=symbols.ne(0).detach()
            rl_mask_2=symbols.ne(2).detach()
            rl_mask_3=symbols.ne(3).detach()
            end_ = end_*rl_mask_2.float()
            prob = combined_vocab.gather(1, symbols).add_(sys.float_info.epsilon)
            probs.append(prob.log().mul(-1)*rl_mask_0.float()*rl_mask_2.float()*rl_mask_3.float()*end_)

            eos_batches = symbols.data.eq(self.eos_id)
            
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > _step) & eos_batches) != 0
                
                lengths[update_idx] = len(decoded_outputs)
            # change unk to corresponding field
            for i in range(symbols.size(0)):
                
                w2f = w2fs[i]
        
                if symbols[i].item() > self.vocab_size-1:
                    symbols[i] = w2f[symbols[i].item()]
            
            embed_input = self.embedding(symbols)
            decoder_hidden = _c
            coverage = coverage + attn_scores
        lens  = torch.FloatTensor(lengths.tolist())
        if self.USE_CUDA:
            lens = lens.cuda()
        cand_probs=torch.cat(probs, 1).sum(1).div(lens).cpu().view(-1).numpy()
        if fig:
            print('go fig')
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), f_matrix[0], \
                   torch.stack(attn, 1).squeeze(2)[0]
        else:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), cand_probs

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            h = self.Wh(h)
        return h

    def _validate_args(self, targets, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        # encoder_outputs -> value
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        else:
            max_enc_len = encoder_outputs.size(1)
        # inference batch size
        if targets is None and encoder_hidden is None:
            batch_size = 1
        else:
            if targets is not None:
                batch_size = targets.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default targets and max decoding length
        if targets is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")
            # torch.set_grad_enabled(False)
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if self.USE_CUDA:
                targets = targets.cuda()
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        # max_enc_len: max length of value
        return targets, batch_size, max_length, max_enc_len










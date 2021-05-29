import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys

def _inflate(tensor, times, dim):
        """
        Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
        Args:
            tensor: A :class:`Tensor` to inflate
            times: number of repetitions
            dim: axis for inflation (default=0)
        Returns:
            A :class:`Tensor`
        Examples::
            >> a = torch.LongTensor([[1, 2], [3, 4]])
            >> a
            1   2
            3   4
            [torch.LongTensor of size 2x2]
            >> b = ._inflate(a, 2, dim=1)
            >> b
            1   2   1   2
            3   4   3   4
            [torch.LongTensor of size 2x4]
            >> c = _inflate(a, 2, dim=0)
            >> c
            1   2
            3   4
            1   2
            3   4
            [torch.LongTensor of size 4x2]
        """
        repeat_dims = [1] * tensor.dim()
        repeat_dims[dim] = times
        return tensor.repeat(*repeat_dims)

class TopKDecoder(torch.nn.Module):
    r"""
    Top-K decoding with beam search.
    Args:
        decoder_rnn (DecoderRNN): An object of DecoderRNN used for decoding.
        k (int): Size of the beam.
    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (seq_len, batch, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features
          in the hidden state `h` of encoder. Used as the initial hidden state of the decoder.
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*length* : list of integers
          representing lengths of output sequences, *topk_length*: list of integers representing lengths of beam search
          sequences, *sequence* : list of sequences, where each sequence is a list of predicted token IDs,
          *topk_sequence* : list of beam search sequences, each beam is a list of token IDs, *inputs* : target
          outputs if provided for decoding}.
    """

    def __init__(self, decoder_rnn, k):
        super(TopKDecoder, self).__init__()
        self.rnn = decoder_rnn
        self.k = k
        self.hidden_size = self.rnn.hidden_size
        self.V = self.rnn.output_size
        self.SOS = self.rnn.sos_id
        self.EOS = self.rnn.eos_id
        self.UNK = self.rnn.unk_id

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_mask=None, encoder_hidden=None, encoderi=None, encoderf=None,
                encoderp=None, teacher_forcing_ratio=None,train_mode = None,reward_cal = None,list_oovs=None,weights=None,w2fs=None,fig=False,batch_cand_rewards = None,norm_max_min=None,retain_output_probs=True):
        """
        Forward rnn for MAX_LENGTH steps.  Look at :func:`seq2seq.models.DecoderRNN.DecoderRNN.forward_rnn` for details.
        """

        # inputs, batch_size, max_length = self.rnn._validate_args(inputs, encoder_hidden, encoder_outputs,
        #                                                          function, teacher_forcing_ratio)
        self.V = self.rnn.output_size + max_source_oov
        targets, batch_size, max_length, max_enc_len = self.rnn._validate_args(targets, encoder_hidden, encoderi, teacher_forcing_ratio)
        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)
        if self.rnn.USE_CUDA:
            self.pos_index = self.pos_index.cuda()
        # Inflate the initial hidden states to be of size: b*k x h
        encoder_hidden = self.rnn._init_state(encoder_hidden)
        if encoder_hidden is None:
            hidden = None
        else:
            if isinstance(encoder_hidden, tuple):
                hidden = tuple([torch.repeat_interleave(h, self.k, 1) for h in encoder_hidden])
            else:
                hidden = torch.repeat_interleave(encoder_hidden, self.k, 1)

        
        coverage = torch.zeros(batch_size, max_enc_len)
        coverage = torch.repeat_interleave(coverage, self.k, 0)
        
        enci_proj = self.rnn.Wih(encoderi.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        encf_proj = self.rnn.Wfh(encoderf.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        f_matrix = self.rnn.get_matrix(encoderp)
        enc_proj = enci_proj + encf_proj
        enc_proj = torch.repeat_interleave(enc_proj, self.k, 0)
        # get link attention scores -> S*, V*
        c_contexti, c_contextf = self.rnn.self_attn(f_matrix, encoderi, encoderf)
        c_contexti = torch.repeat_interleave(c_contexti, self.k, 0)
        c_contextf = torch.repeat_interleave(c_contextf, self.k, 0)
        f_matrix = torch.repeat_interleave(f_matrix, self.k, 0)
        input_ids = torch.repeat_interleave(input_ids, self.k , 0)
        enc_mask = torch.repeat_interleave(enc_mask, self.k, 0)
        if self.rnn.USE_CUDA:
            coverage = coverage.cuda()
            
        

        # ... same idea for encoder_outputs and decoder_outputs
        # if self.rnn.use_attention:
        #     inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 0)
        # else:
        #     inflated_encoder_outputs = None
        
        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)
        if self.rnn.USE_CUDA:
            sequence_scores = sequence_scores.cuda()

        # Initialize the input vector
        input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))
        if self.rnn.USE_CUDA:
            input_var = input_var.cuda()
        embed_input = self.rnn.embedding(input_var)
        # Store decisions for backtracking
        stored_outputs = list()  # distribution at each step
        stored_scores = list()    #  scores at each step
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for _step in range(max_length):

            # Run the RNN one step forward
            _h, _c = self.rnn.rnn(embed_input, hidden)
            combined_vocab, attn_scores = self.rnn.decode_step(input_ids, coverage,
                                                           _h.squeeze(1), enc_proj, batch_size, max_enc_len, enc_mask,
                                                           c_contexti, c_contextf, embed_input.squeeze(1),
                                                           max_source_oov, f_matrix)
            log_combined_vocab = combined_vocab.add_(sys.float_info.epsilon).log().clone().detach()
            log_combined_vocab[:, self.UNK] = -float('Inf')

            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_combined_vocab)

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = _inflate(sequence_scores, self.V, 1)
            sequence_scores += log_combined_vocab.squeeze(1)
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)
            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)
            sequence_scores = scores.view(batch_size * self.k, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1).long()
            
            if isinstance(_c, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in _c])
            else:
                hidden = _c.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone().detach())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var.clone().detach())
            stored_hidden.append(hidden)

            # change oov to corresponding field
            
            for i in range(input_var.size(0)):
                w2f = w2fs[i // self.k]
                if input_var[i].item() > self.rnn.output_size - 1:
                    input_var[i] = w2f[input_var[i].item()]

            # update parameters for next timestep
            embed_input = self.rnn.embedding(input_var)
            coverage = coverage + attn_scores


        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_size)

        # Build return objects
        decoder_outputs = torch.transpose(torch.stack(p,0),0,1).squeeze()
        first_index = torch.LongTensor([0])
        if self.rnn.USE_CUDA:
            decoder_outputs = decoder_outputs.cuda()
            first_index = first_index.cuda()
        decoder_outputs = torch.index_select(decoder_outputs,2,first_index)
        seq_lengths = torch.FloatTensor([seq_len[0] for seq_len in l])
        s = torch.stack([score[0] for score in s])
        if self.rnn.USE_CUDA:
            s = s.cuda()
            seq_lengths = seq_lengths.cuda()
        s=s.div(seq_lengths).cpu().numpy()
        return decoder_outputs, [seq_len[0] for seq_len in l], s

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
        if self.rnn.USE_CUDA:
            h_n = h_n.cuda()
        l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    if lstm:
                        current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                        current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                        h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                        h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                    else:
                        current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                        h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for step in reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data

        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
            score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)


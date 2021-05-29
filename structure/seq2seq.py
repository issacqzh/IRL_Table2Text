import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb, input_lengths=None,
                target=None, target_id=None, teacher_forcing_ratio=0,train_mode='sl',reward_cal=None,weights=None,list_oovs=None,w2fs=None, fig=False,batch_cand_rewards = None,norm_max_min=None,extractor_field_mask_tensor=None):

        # encoderi: value, encoderf: field, encoderp: (forward position, backward position)
        # encoder_hidden: hidden state of the last step   
        
        encoderi, encoderf, encoderp, encoder_hidden, mask = self.encoder(batch_s, batch_f, batch_pf, batch_pb,
                                                                        input_lengths)
                
        result= self.decoder(max_source_oov=max_source_oov,
                              targets=target,
                              targets_id=target_id,
                              input_ids=batch_o_s,
                              enc_mask=mask,
                              encoder_hidden=encoder_hidden,
                              encoderi=encoderi,
                              encoderf=encoderf,
                              encoderp=encoderp,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              train_mode = train_mode,
                              reward_cal = reward_cal,
                              list_oovs=list_oovs,
                              weights = weights,
                              w2fs=w2fs,
                              fig=fig,
                              batch_cand_rewards=batch_cand_rewards,
                              norm_max_min=norm_max_min)
        
        return result




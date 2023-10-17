from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
class NMTConfig:
    def __init__(self, embed_size=512, vocab = None, hidden_size=512, dropout_rate = 0.2) -> None:
        self.embed_size = embed_size
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])   

class NMT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        #self.model_embeddings = ModelEmbeddings(config.embed_size, vocab)
        self.config = config
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.vocab = config.vocab
        self.embed_size = config.embed_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.src_embedding = nn.Embedding(len(self.vocab.src), self.embed_size, padding_idx=self.vocab.src['<pad>'])
        self.tgt_embedding = nn.Embedding(len(self.vocab.tgt), self.embed_size, padding_idx=self.vocab.tgt['<pad>'])
        #self.post_embed_cnn = nn.Conv1d(in_channels=config.embed_size,out_channels=config.embed_size, kernel_size=2,padding="same",device=self.device)
        
        self.encoder = nn.LSTM(input_size = config.embed_size, hidden_size = self.hidden_size, bias = True, bidirectional = True,device=self.device) # default bias = True
        self.decoder = nn.LSTMCell(input_size = config.embed_size+self.hidden_size, hidden_size = self.hidden_size, bias = True, device=self.device)

        self.h_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False,device=self.device)
        self.c_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False,device=self.device)
        self.att_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False,device=self.device)
        self.combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False,device=self.device)

        self.target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.tgt), bias=False,device=self.device)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, source, target) -> torch.Tensor:
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)    # enc_hiddens: (b,src_len, 2*h), dec_init_state = (h_0^{dec}, c_0^{dec}): both (b,h)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)           # enc_masks: (b, src_len), tensor with 1 for padding tokens otherwise 0
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)   # combined_outputs: (target_len, b, h)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)               # P: (target_len, b, len(vocab.tgt))
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()                       # target_masks: (target_len, b)
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]      # (target_len, b)
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores   # scores: (b)
    
    def encode(self, source_padded: torch.Tensor, source_lengths: List[str]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        X = self.src_embedding(source_padded)                             # (src_len, b, e)
        #X = torch.permute(X, (1,2,0))                                               # (b,e,src_len)
        #X = self.post_embed_cnn(X)                                                  # (b,e,src_len)
        #X = torch.permute(X, (2,0,1))                                               # (src_len,b,e)
        src_len = X.size()[0]                                                       # useful as total_length in pad_packed_sequence 
        X_packed = pack_padded_sequence(X, torch.Tensor(source_lengths))
        enc_hiddens, (last_hidden, last_cell)  = self.encoder(X_packed)             # enc_hiddens: (src_len, b, 2*h), last_hidden: (2, b, h)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, total_length=src_len)     # enc_hiddens: (src_len, b, 2*h)
        enc_hiddens = torch.permute(enc_hiddens, (1,0,2))                           # enc_hiddens: (b, src_len, 2*h)
        last_hidden = torch.cat((last_hidden[0],last_hidden[1]), dim=-1)            # (b, 2*h)
        init_decoder_hidden = self.h_projection(last_hidden)                        # (b, h)
        last_cell = torch.cat((last_cell[0],last_cell[1]), dim=-1)                  # (b, 2*h)
        init_decoder_cell = self.h_projection(last_cell)                            # (b, h)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        return enc_hiddens, dec_init_state
    
    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor, dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        target_padded = target_padded[:-1]             # for deleting the <END> token
        dec_state = dec_init_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []
        enc_hiddens_proj = self.att_projection(enc_hiddens)     # (b, src_len, h)
        Y = self.tgt_embedding(target_padded)         # (tgt_len, b, e)
        for Y_t in torch.split(Y,split_size_or_sections=1):     # Y_t: (1, b, e)
            Y_t = Y_t.squeeze(0)                                # (b, e)
            Ybar_t = torch.cat((Y_t, o_prev), dim=-1)           # (b, e+h)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack(combined_outputs)         # (tgt_len, b, h)
        return combined_outputs
    
    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        
        dec_state = self.decoder(Ybar_t, dec_state)                # (b, h)
        dec_hidden, dec_cell = dec_state
        e_t = torch.bmm(enc_hiddens_proj,dec_hidden.unsqueeze(-1)).squeeze(-1)      # (b, src_len)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
        alpha_t = F.softmax(e_t,dim=-1)                                             # (b, src_len)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)               # (b, 2h)
        u_t = torch.cat((a_t, dec_hidden),-1)                                       # (b, 3h)
        v_t = self.combined_output_projection(u_t)                                  # (b, h)
        o_t = self.dropout(torch.tanh(v_t))                                         # (b, h)
        return dec_state, o_t, e_t

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[Hypothesis]:
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)
        eos_id = self.vocab.tgt['</s>']
        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []
        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)
            (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)
            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()
                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)
            if len(completed_hypotheses) == beam_size:
                break
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]
            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],score=hyp_scores[0].item()))
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses
    


    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        config = params['config']
        model = NMT(config)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

if __name__=='__main__':
    from vocab import *
    vocab = Vocab.load('vocab.json')
    config = NMTConfig(embed_size = 256, vocab = vocab)
    model = NMT(config)
    print(model)


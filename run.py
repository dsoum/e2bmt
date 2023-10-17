
"""
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --valid-number=<int>                    number of validation in training [default: 5]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
# import pickle
import time


from docopt import docopt
import numpy as np
from typing import List, Tuple, Dict, Set, Union

import torch
import torch.nn.utils
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

from nmt_model import Hypothesis, NMT, NMTConfig
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
# import sacrebleu

def train(args):
    train_data_src = read_corpus(args['--train-src'], source='src', vocab_size=21000)       # EDIT: NEW VOCAB SIZE
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt', vocab_size=8000)

    dev_data_src = read_corpus(args['--dev-src'], source='src', vocab_size=3000)
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt', vocab_size=2000)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])
    config = NMTConfig(embed_size=int(args['--embed-size']),                                 # EDIT: 4X EMBED AND HIDDEN SIZES 
                       hidden_size=int(args['--hidden-size']),
                       dropout_rate=float(args['--dropout']),
                       vocab=vocab)
    model = NMT(config)
    model.train()
    #model initialization
    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    epoch = 0
    print('begin Maximum Likelihood training')
    print(args['--max-epoch'], type(args['--max-epoch']))
    while epoch< int(args['--max-epoch']):
        epoch += 1
        cum_loss, cum_size = 0,0
        model.train()
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            optimizer.zero_grad()
            batch_size = len(src_sents)
            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size
            loss.backward()
            optimizer.step()
            # batch_losses_val = batch_loss.item()
            cum_loss += batch_loss.item()
            cum_size += batch_size
        train_loss = cum_loss/cum_size
        
        model.eval()
        cum_loss, cum_size = 0, 0
        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = -model(src_sents, tgt_sents).sum()
                cum_loss += loss.item()
                cum_size += len(src_sents)
            val_loss = cum_loss/cum_size
        print(f"epoch: {epoch}, train_loss loss: {train_loss}, validation_loss: {val_loss}")
        model.save(model_save_path)
            

def decode(args):
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    model = NMT.load(args['MODEL_PATH'])
    model = model.to(device)

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),            
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ''.join(top_hyp.value).replace('▁', ' ')
            f.write(hyp_sent + '\n')

def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    was_training = model.training
    model.eval()
    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
            
    if was_training: model.train(was_training)
    return hypotheses

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]
    bleu = corpus_bleu(detokened_hyps, [detokened_refs])
    return bleu.score

def main():
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__=="__main__":
    main()
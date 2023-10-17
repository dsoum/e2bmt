import sentencepiece as spm
import math
import numpy as np


def pad_sents(sents, pad_token):
    sents_padded = []
    max_len = max([len(s) for s in sents])
    for s in sents:
        sents_padded.append(s + [pad_token]*(max_len-len(s)))
    
    return sents_padded

def read_corpus(file_path, source, vocab_size):
    data = []
    sp = spm.SentencePieceProcessor()
    sp.Load('{}.model'.format(source))

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)
    return data

def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data)/batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
    
    for i in range(batch_num):
        indices = index_array[i*batch_size: (i+1)*batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key = lambda x: len(x[0]), reverse = True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        yield src_sents, tgt_sents
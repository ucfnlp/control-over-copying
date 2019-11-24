import argparse
import json
import pickle
import random
import re
import shutil
from copy import deepcopy as cp

import numpy as np
import torch
import torch.nn.functional as F


# IO
def loadFromJson(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f, strict=False)
    f.close()
    return data


def saveToJson(filename, data):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True


def saveToPKL(filename, data):
    with open(filename, 'wb')as f:
        pickle.dump(data, f)
    return


def loadFromPKL(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def writeFile(filename, massage):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(massage)
    return True


# Initializing
def zero_weights(n_in, n_out=None):
    if (n_out == None):
        W = np.zeros(n_in)
    else:
        W = np.zeros(n_in, n_out)
    return W.astype('float32')


def orthogonal_weights(n_dim):
    W = np.random.randn(n_dim, n_dim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')


def random_weights(n_in, n_out, scale=None):
    if scale is None:
        scale = np.sqrt(2.0 / (n_in + n_out))
    W = scale * np.random.randn(n_in, n_out)
    return W.astype('float32')


def remove_digits(parse):
    return re.sub(r'\d', '#', parse)


def save_check_point(state, is_best, path='.model', fileName='latest.pth.tar'):
    torch.save(state, path + '/' + fileName)
    if is_best:
        shutil.copyfile(path + '/' + fileName, path + '/model_best.pth.tar')
        shutil.copyfile(path + '/' + fileName, path + '/model_best_epoch_' + str(state['epoch']) + '.pth.tar')


def RougeTrick(parse):
    '''
    parse = re.sub(r'#','XXX',parse)
    parse = re.sub(r'XXX-','XXXYYY',parse)
    parse = re.sub(r'-XXX','YYYXXX',parse)
    parse = re.sub(r'XXX.','XXXWWW',parse)
    parse = re.sub(r'.XXX','WWWXXX',parse)
    parse = re.sub(r'<unk>','ZZZZZ',parse)
    '''
    parse = re.sub(r'#', 'T', parse)
    parse = re.sub(r'T-', 'TD', parse)
    parse = re.sub(r'-T', 'DT', parse)
    parse = re.sub(r'TX.', 'TB', parse)
    parse = re.sub(r'.T', 'BT', parse)
    parse = re.sub(r'<unk>', 'UNK', parse)

    return parse


def from_dict(json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = argparse.Namespace()
    for key, value in json_object.items():
        config.__dict__[key] = value
    return config


def from_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, "r", encoding='utf-8') as reader:
        text = reader.read()
    return from_dict(json.loads(text))


def Index2Word(Index, Vocab):
    return Vocab.i2w[Index]


def Word2Index(Word, Vocab):
    if (not Word in Vocab.w2i):
        Word = '<unk>'
    return Vocab.w2i[Word]


def Sentence2ListOfWord(sentence):
    listOfWord = sentence.split()
    return listOfWord


def ListOfWord2ListOfIndex(listOfWord, Vocab):
    listOfIndex = []
    for w in listOfWord:
        listOfIndex.append(Word2Index(w, Vocab))
    return listOfIndex


def Sentence2ListOfIndex(sentence, Vocab):
    return ListOfWord2ListOfIndex(Sentence2ListOfWord(sentence), Vocab)


def maskDropout_col(x, r, mask):
    mc = x.data.new(1, x.size(1)).bernoulli_(r)
    return x * (1 - mc * mask)


def unimask(n, m):
    mask = torch.zeros(n + m, n + m)
    mask[:n, :n] = torch.ones(n, n)
    mask[n:, :n] = torch.ones(m, n)
    mask[n:, n:] = torch.tril(torch.ones(m, m))
    return mask


def maskSeq_new(seq, prob, config, onlyOn=None):
    length = len(seq)
    seq_input = []
    seq_label = []
    for i in range(length):
        rrd = random.random()
        if ((onlyOn is None) or (onlyOn[i] > 0)) and (rrd <= prob):
            seq_label.append(seq[i])
            rd = random.random()
            if rd <= 0.8:
                seq_input.append(config.MASK)
            elif rd <= 0.9:
                seq_input.append(seq[i])
            else:
                seq_input.append(random.choice(list(range(config.n_vocab))))
        else:
            seq_label.append(config.PAD)
            seq_input.append(seq[i])
    return seq_input, seq_label


def cos_similarity(x, y, embedding):
    return F.cosine_similarity(embedding(x).unsqueeze(1), embedding(y).unsqueeze(0), dim=2)


def gen_loss_mask(src, tgt, para, eps=1e-6):
    scores = cos_similarity(torch.LongTensor(tgt), torch.LongTensor(src), para['embedding']).max(1)[0]
    mask = []
    this_seen, this_similar, this_unseen = 0, 0, 0
    for score in scores:
        r = random.random()
        if score >= 1 - eps:
            if (r < para['seen_prob']):
                mask.append(1)
                this_seen += 1
            else:
                mask.append(0)
        elif score >= para['similar_threshold']:
            if (r < para['similar_prob']):
                mask.append(1)
                this_similar += 1
            else:
                mask.append(0)
        else:
            if (r < para['unseen_prob']):
                mask.append(1)
                this_unseen += 1
            else:
                mask.append(0)
    return mask + [1], this_seen, this_similar, this_unseen


def prepare_data(srcs, tgts, config, prob_src=0.1, prob_tgt=0.9,
                 para={"seen_prob": 1.0, "similar_prob": 1.0, "unseen_prob": 1.0, "similar_threshold": 0.7,
                       "embedding": None}):
    ls = [len(src) + len(tgt) + 3 for src, tgt in zip(srcs, tgts)]
    l_max = max(ls)

    padded_inputs = []
    padded_positions = []
    padded_token_types = []

    padded_labels = []
    padded_masks = []

    total_seen = 0
    total_similar = 0
    total_unseen = 0
    total_source = 0
    for src, tgt, l in zip(srcs, tgts, ls):
        src_ = [config.CLS] + src + [config.SEP]
        tgt_ = tgt + [config.SEP]

        tgt_loss_mask, tgt_seen, tgt_similar, tgt_unseen = gen_loss_mask(src, tgt, para)

        # Mask Both Sides
        new_src_input, new_src_label = maskSeq_new(src_, prob_src, config)
        new_tgt_input, new_tgt_label = maskSeq_new(tgt_, prob_tgt, config)

        # Data Modification of seen and unseen words
        new_tgt_label = [it[0] * it[1] for it in zip(new_tgt_label, tgt_loss_mask)]
        total_seen += tgt_seen
        total_similar += tgt_similar
        total_unseen += tgt_unseen
        total_source += sum([int(label == 0) for label in new_src_label])

        # Cast to Torch Types.

        input_i = torch.LongTensor(new_src_input + new_tgt_input)
        position_i = torch.cat([torch.arange(len(src_)), torch.arange(1, len(tgt_) + 1)]).long()
        token_type_i = torch.LongTensor([0] * (len(src_)) + [1] * len(tgt_))
        label_i = torch.LongTensor(new_src_label + new_tgt_label)

        mask_i = unimask(len(src_), len(tgt_))

        # Padding
        padded_input_i = F.pad(input_i, (0, l_max - l), "constant", 0)
        padded_position_i = F.pad(position_i, (0, l_max - l), "constant", 0)
        padded_token_type_i = F.pad(token_type_i, (0, l_max - l), "constant", 0)

        padded_label_i = F.pad(label_i, (0, l_max - l), "constant", 0)
        padded_mask_i = F.pad(mask_i, (0, l_max - l, 0, l_max - l), "constant", 0.0)

        # Append to list

        padded_inputs.append(padded_input_i)
        padded_positions.append(padded_position_i)
        padded_token_types.append(padded_token_type_i)

        padded_labels.append(padded_label_i)
        padded_masks.append(padded_mask_i)

    inputs, positions, token_types, labels, masks = torch.stack(padded_inputs), \
                                                    torch.stack(padded_positions), torch.stack(padded_token_types), \
                                                    torch.stack(padded_labels), torch.stack(padded_masks)
    '''
    if not config.dataParallel:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            positions = positions.cuda()
            token_types = token_types.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
    '''
    return [inputs, positions, token_types, labels, masks, total_seen, total_similar, total_unseen, total_source]


def prepare_test_data(src, tgt, config):
    src = [config.CLS] + src + [config.SEP]
    tgt = tgt + [config.MASK]

    inputs = torch.LongTensor(src + tgt)
    positions = torch.cat([torch.arange(len(src)), torch.arange(1, len(tgt) + 1)]).long()
    token_types = torch.LongTensor([0] * len(src) + [1] * len(tgt))
    masks = unimask(len(src), len(tgt))

    inputs, positions, token_types, masks = inputs.unsqueeze(0), positions.unsqueeze(0), token_types.unsqueeze(
        0), masks.unsqueeze(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda(torch.cuda.current_device())
        positions = positions.cuda(torch.cuda.current_device())
        token_types = token_types.cuda(torch.cuda.current_device())
        masks = masks.cuda(torch.cuda.current_device())

    return [inputs, positions, token_types, masks]


def nGram(seq, n):
    return list(zip(*[seq[i:] for i in range(n)]))


def do_tricks(preds, source, target, config):
    ban_ids = []
    if config.triGramTrick and len(target) > 2:
        current_triGrams = nGram(target, 3)
        for triGram in current_triGrams:
            if (target[-2] == triGram[0]) and (target[-1] == triGram[1]):
                ban_ids.append(triGram[2])

    ratio = 0.0
    bonus_ids = []
    if config.biGramTrick and len(target) > 0:
        bi_in = set(nGram(source, 2))
        bi_now = set(nGram(target, 2))
        available_biGrams = bi_in - bi_now

        for biGram in list(available_biGrams):
            if (target[-1] == biGram[0]):
                bonus_ids.append(biGram[1])

        ratio = config.gamma_value / (len(bi_in) + 1e-8)

    for idx in bonus_ids:
        # preds[idx] = min(0, preds[idx] + ratio)
        preds[idx] += ratio

    for idx in ban_ids:
        preds[idx] = -1e9

    return preds


def format_(x, y):
    fx = open(x, 'r')
    fy = open(y, 'w')
    for l in fx:
        line = l.lower().strip()
        print(line, file=fy)
    fx.close()
    fy.close()


def mapping_tokenize(s, t):
    st = 0
    ed = 0
    mapping = []
    mapping_idx = []
    for idx, token in enumerate(s):
        token_ = token.lower()
        prefix = "".join([piece.replace('##', '') for piece in t[st:ed + 1]])
        # print(prefix, type(prefix))
        while token_.startswith(prefix):
            ed += 1
            if ed >= len(t):
                break
            prefix = "".join([piece.replace('##', '') for piece in t[st:ed + 1]])
            # print(prefix, type(prefix))
        if (ed - st > 1) or (sum(1 for c in token if c.isupper()) > 1) or (idx > 0):
            mapping_idx.append([(st, ed), idx])
            mapping.append([cp(t[st:ed]), token])
        st = ed
    return mapping


def detokenize(text, mapping):
    text = " " + text
    for one_mapping in mapping:
        keys = "".join([key.replace('##', '') if key.startswith('##') else ' ' + key for key in one_mapping[0]])
        value = ' ' + one_mapping[1]
        text = text.replace(keys, value)
    text = list(text[1:])
    if len(text) > 0:
        text[0] = text[0].upper()
        text = "".join(text)
    return text

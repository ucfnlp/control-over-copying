import math
from copy import deepcopy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertLayerNorm, BertSelfOutput, BertOutput, BertIntermediate, BertPooler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertLMPredictionHead
from scipy import signal
from torch.autograd import Variable


def clones(module, N):
    return nn.ModuleList([cp(module) for _ in range(N)])


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
                      :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs_)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attns = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attns


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attns = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attns


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = clones(layer, config.num_hidden_layers)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        outputs = []
        attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(
                hidden_states, attention_mask)
            if output_all_encoded_layers:
                outputs.append(hidden_states)
                attentions.append(attention)
        if not output_all_encoded_layers:
            outputs.append(hidden_states)
            attentions.append(attention)
        return outputs, attentions


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask, output_all_encoded_layers=True):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, position_ids, token_type_ids)
        outputs, attentions = self.encoder(embedding_output,
                                           extended_attention_mask,
                                           output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = outputs[-1]
        pooled_output = self.pooler(sequence_output)
        return outputs, pooled_output, attentions


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return F.log_softmax(prediction_scores, dim=-1)


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask, masked_lm_labels=None,
                outputAttns=False):
        sequence_output, _, attentions = self.bert(input_ids, position_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=True)
        prediction_scores = self.cls(sequence_output[-1])
        if outputAttns:
            return prediction_scores, attentions
        return prediction_scores

    def similarity(self, x, y):
        '''
        emb_x: N * D
        emb_y: N * D
        score = emb_x * emb_y^T
        length = (emb_x * emb_x).sum(dim = 1)
        '''
        emb_x = self.bert.embeddings.word_embeddings(x)
        emb_y = self.bert.embeddings.word_embeddings(y)
        length_x = torch.sqrt((emb_x * emb_x).sum(dim=1, keepdim=True))
        length_y = torch.sqrt((emb_y * emb_y).sum(dim=1, keepdim=True))
        emb_x /= length_x
        emb_y /= length_y
        score = torch.matmul(emb_x, emb_y.transpose(0, 1))
        return torch.relu(score)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, position_ids, token_type_ids, attention_mask,
                                        output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        predicts = F.log_softmax(self.classifier(pooled_output))

        return predicts


class LabelSmoothing(nn.Module):
    def __init__(self, config):
        super(LabelSmoothing, self).__init__()
        self.crit = nn.KLDivLoss(size_average=False)
        self.pad_idx = config.pad_idx
        self.confidence = 1.0 - config.label_smoothing
        self.smoothing = config.label_smoothing
        self.size = config.n_vocab

    def forward(self, predicts, target):
        assert self.size == predicts.size(1)
        dist = torch.full_like(predicts, self.smoothing / (self.size - 2))
        dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        dist[:, self.pad_idx] = 0
        mask_idx = torch.nonzero(target.data == self.pad_idx)
        if mask_idx.dim() > 0:
            dist.index_fill_(0, mask_idx.squeeze(), 0.0)
        return self.crit(predicts, Variable(dist, requires_grad=False))


class KLDivLoss(nn.Module):
    def __init__(self, config):
        super(KLDivLoss, self).__init__()
        self.crit = LabelSmoothing(config)

    def forward(self, predicts, target, norm=1.0):
        loss = self.crit(predicts.contiguous().view(-1,
                                                    predicts.size(-1)), target.contiguous().view(-1))
        return loss / norm


def interp_(x, xp, fp):
    L = list(xp.size())[0]
    cp_ = (x.reshape(-1, 1) - xp) >= 0
    cp_ = torch.cat([torch.ones_like(cp_[:, :1]).byte(), cp_,
                     torch.zeros_like(cp_[:, :1]).byte()], dim=-1)
    idx = torch.max(cp_[:, :-1] ^ cp_[:, 1:], dim=-1)[1] - 1
    idx = torch.where(idx >= 0, idx, torch.zeros_like(idx))
    idx = torch.where(idx <= L - 2, idx, torch.full_like(idx, L - 2))
    st = xp[idx]
    ed = xp[idx + 1]
    w = (x - st) / (ed - st)
    result = torch.lerp(fp[idx], fp[idx + 1], w)
    result[0] = 0.0
    return result


def pWasserstein_(I0, I1, p=2.0):
    print(I0.size(), I1.size())
    assert I0.size() == I1.size()
    eps = 1e-7
    I0 += eps
    I1 += eps
    I0 = I0 / I0.sum()
    I1 = I1 / I1.sum()
    J0 = I0.cumsum(dim=0)
    J1 = I1.cumsum(dim=0)
    L = list(I0.size())[0]
    x = torch.arange(L).float()
    x = x.to(device=I0.device)
    xtilde = torch.linspace(0, 1, L)
    xtilde = xtilde.to(device=I0.device)
    XI0 = interp_(xtilde, J0, x)
    XI1 = interp_(xtilde, J1, x)
    u = interp_(x, XI0, XI0 - XI1)
    Wp = Wp = (((u.abs() ** p) * I0).mean()) ** (1.0 / p)
    return Wp


class pWasserstein(nn.Module):
    def __init__(self, p=2.0):
        super(pWasserstein, self).__init__()
        self.p = p

    def forward(self, inputs, target):
        return pWasserstein_(inputs, target, self.p)


class GaussianSmoothing(nn.Module):
    def __init__(self, config):
        super(GaussianSmoothing, self).__init__()
        # self.crit = pWasserstein(p=2.0)
        self.crit = nn.KLDivLoss(size_average=False)
        self.kernel = torch.Tensor(signal.gaussian(9, std=1)).unsqueeze(0).unsqueeze(1)
        self.size = config.gen_max_len

    def forward(self, predicts, target, norm=1.0):
        assert self.size == predicts.size(1)
        dist = torch.zeros_like(predicts)
        dist = dist.scatter(1, target.data.unsqueeze(1), 1.0).unsqueeze(1)
        kernel = cp(self.kernel).to(device=target.device)

        dist = F.conv1d(dist, kernel, padding=(4,)).squeeze(1)
        dist = Variable(dist, requires_grad=False).to(device=target.device)
        # print(predicts.size(), dist.size())

        # loss = 0.0
        # N = int(predicts.size()[0])
        # for i in range(N):
        # loss += self.crit(predicts[i], dist[i])

        return self.crit(predicts, dist) / norm

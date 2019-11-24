import time, random, gc, os, math, argparse
from os import path
import numpy as np

import torch
import torch.utils.data as data

from mylog import mylog
from options_process import optionsLoader
from data_process import myDataSet_Bert as Dataset
from utility import *
from model_bert import *
from searcher import Searcher
from searcher.scorer import *

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from parallel import DataParallelModel, DataParallelCriterion

random.seed(time.time())
LOG = mylog(reset=True)

def train(config):
    net = BertForMaskedLM.from_pretrained(config.model)
    lossFunc = KLDivLoss(config)

    if torch.cuda.is_available():
        net = net.cuda()
        lossFunc = lossFunc.cuda()
        
        if config.dataParallel:
            net = DataParallelModel(net)
            lossFunc = DataParallelCriterion(lossFunc)

    options = optionsLoader(LOG, config.optionFrames, disp=False)
    Tokenizer = BertTokenizer.from_pretrained(config.model)
    prepareFunc = prepare_data

    trainSet = Dataset('train', config.batch_size, lambda x: len(x[0]) + len(x[1]), prepareFunc, Tokenizer,
                       options['dataset'], LOG, 'train')
    validSet = Dataset('valid', config.batch_size, lambda x: len(x[0]) + len(x[1]), prepareFunc, Tokenizer,
                       options['dataset'], LOG, 'valid')

    print(trainSet.__len__())

    Q = []
    best_vloss = 1e99
    counter = 0
    lRate = config.lRate

    prob_src = config.prob_src
    prob_tgt = config.prob_tgt


    num_train_optimization_steps = trainSet.__len__() * options['training']['stopConditions']['max_epoch']
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lRate,
                         e=1e-9,
                         t_total=num_train_optimization_steps,
                         warmup=0.0)

    for epoch_idx in range(options['training']['stopConditions']['max_epoch']):
        total_seen = 0
        total_similar = 0
        total_unseen = 0
        total_source = 0

        trainSet.setConfig(config, prob_src, prob_tgt)
        trainLoader = data.DataLoader(
            dataset=trainSet,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataLoader_workers,
            pin_memory=True
        )

        validSet.setConfig(config, 0.0, prob_tgt)
        validLoader = data.DataLoader(
            dataset=validSet,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataLoader_workers,
            pin_memory=True
        )

        for batch_idx, batch_data in enumerate(trainLoader):
            if (batch_idx + 1) % 10000 == 0:
                gc.collect()
            start_time = time.time()

            net.train()

            inputs, positions, token_types, labels, masks, batch_seen, batch_similar, batch_unseen, batch_source = batch_data

            inputs = inputs[0].cuda()
            positions = positions[0].cuda()
            token_types = token_types[0].cuda()
            labels = labels[0].cuda()
            masks = masks[0].cuda()
            total_seen += batch_seen
            total_similar += batch_similar
            total_unseen += batch_unseen
            total_source += batch_source
            
            
            n_token = int((labels.data != 0).data.sum())

            predicts = net(inputs, positions, token_types, masks)
            loss = lossFunc(predicts, labels, n_token).sum()

            Q.append(float(loss))
            if len(Q) > 200:
                Q.pop(0)
            loss_avg = sum(Q) / len(Q)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            LOG.log('Epoch %2d, Batch %6d, Loss %9.6f, Average Loss %9.6f, Time %9.6f' % (
            epoch_idx + 1, batch_idx + 1, loss, loss_avg, time.time() - start_time))

            # Checkpoints
            idx = epoch_idx * trainSet.__len__() + batch_idx + 1
            if (idx >= options['training']['checkingPoints']['checkMin']) and (
                    idx % options['training']['checkingPoints']['checkFreq'] == 0):
                if config.do_eval:
                    vloss = 0
                    total_tokens = 0
                    for bid, batch_data in enumerate(validLoader):
                        inputs, positions, token_types, labels, masks, batch_seen, batch_similar, batch_unseen, batch_source = batch_data

                        inputs = inputs[0].cuda()
                        positions = positions[0].cuda()
                        token_types = token_types[0].cuda()
                        labels = labels[0].cuda()
                        masks = masks[0].cuda()

                        n_token = int((labels.data != config.PAD).data.sum())

                        with torch.no_grad():
                            net.eval()
                            predicts = net(inputs, positions, token_types, masks)
                            vloss += float(lossFunc(predicts, labels).sum())

                        total_tokens += n_token

                    vloss /= total_tokens
                    is_best = vloss < best_vloss
                    best_vloss = min(vloss, best_vloss)
                    LOG.log('CheckPoint: Validation Loss %11.8f, Best Loss %11.8f' % (vloss, best_vloss))

                    if is_best:
                        LOG.log('Best Model Updated')
                        save_check_point({
                            'epoch': epoch_idx + 1,
                            'batch': batch_idx + 1,
                            'options': options,
                            'config': config,
                            'state_dict': net.state_dict(),
                            'best_vloss': best_vloss},
                            is_best,
                            path=config.save_path,
                            fileName='latest.pth.tar'
                        )
                        counter = 0
                    else:
                        counter += options['training']['checkingPoints']['checkFreq']
                        if counter >= options['training']['stopConditions']['rateReduce_bound']:
                            counter = 0
                            for param_group in optimizer.param_groups:
                                lr_ = param_group['lr']
                                param_group['lr'] *= 0.55
                                _lr = param_group['lr']
                            LOG.log('Reduce Learning Rate from %11.8f to %11.8f' % (lr_, _lr))
                        LOG.log('Current Counter = %d' % (counter))

                else:
                    save_check_point({
                        'epoch': epoch_idx + 1,
                        'batch': batch_idx + 1,
                        'options': options,
                        'config': config,
                        'state_dict': net.state_dict(),
                        'best_vloss': 1e99},
                        False,
                        path=config.save_path,
                        fileName='checkpoint_Epoch' + str(epoch_idx + 1) + '_Batch' + str(batch_idx + 1) + '.pth.tar'
                    )
                    LOG.log('CheckPoint Saved!')
            
        
        if options['training']['checkingPoints']['everyEpoch']:
            save_check_point({
                'epoch': epoch_idx + 1,
                'batch': batch_idx + 1,
                'options': options,
                'config': config,
                'state_dict': net.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                fileName='checkpoint_Epoch' + str(epoch_idx + 1) + '.pth.tar'
            )
        
        LOG.log('Epoch Finished.')
        LOG.log('Total Seen: %d, Total Unseen: %d, Total Similar: %d, Total Source: %d.' % (total_seen, total_unseen, total_similar, total_source))
        gc.collect()

def translate(Answers, Tokenizer):
    tokens = Tokenizer.convert_ids_to_tokens(Answers[:-1])
    tokens = [token.replace('##', '') if token.startswith('##') else ' ' + token for token in tokens]
    return "".join(tokens)[1:]

def test(config):
    Best_Model = torch.load(config.test_model)
    Tokenizer = BertTokenizer.from_pretrained(config.model)

    f_in = open(config.inputFile, 'r')
    
    net = BertForMaskedLM.from_pretrained(config.model)

    # When loading from a model not trained from DataParallel
    #net.load_state_dict(Best_Model['state_dict'])
    #net.eval()
    
    if torch.cuda.is_available():
        net = net.cuda(0)
        if config.dataParallel:
            net = DataParallelModel(net)

    # When loading from a model trained from DataParallel
    net.load_state_dict(Best_Model['state_dict'])
    net.eval()

    mySearcher = Searcher(net, config)
    
    f_top1 = open('summary' + config.suffix + '.txt', 'w', encoding='utf-8')
    f_topK = open('summary' + config.suffix + '.txt.' + str(config.answer_size), 'w', encoding='utf-8')

    ed = '\n------------------------\n'

    for idx, line in enumerate(f_in):
        source_ = line.strip().split()
        source = Tokenizer.tokenize(line.strip())
        mapping = mapping_tokenize(source_, source)

        
        source = Tokenizer.convert_tokens_to_ids(source)
        
        print(idx)
        print(detokenize(translate(source, Tokenizer), mapping), end=ed)
        
        l_pred = mySearcher.length_Predict(source)
        Answers = mySearcher.search(source)
        baseline = sum(Answers[0][0])
        
        if config.reranking_method == 'none':
            Answers = sorted(Answers, key=lambda x: sum(x[0]))
        elif config.reranking_method == 'length_norm':
            Answers = sorted(Answers, key=lambda x: length_norm(x[0]))
        elif config.reranking_method == 'bounded_word_reward':
            Answers = sorted(Answers, key=lambda x: bounded_word_reward(x[0], config.reward, l_pred))
        elif config.reranking_method == 'bounded_adaptive_reward':
            Answers = sorted(Answers, key=lambda x: bounded_adaptive_reward(x[0], x[2], l_pred))
        
        texts = [detokenize(translate(Answers[k][1], Tokenizer), mapping) for k in range(len(Answers))]
        
        if baseline != sum(Answers[0][0]):
            print('Reranked!')

        print(texts[0], end=ed)
        print(texts[0], file=f_top1)
        print(len(texts), file=f_topK)
        for i in range(len(texts)):
            print(Answers[i][0], file=f_topK)
            print(texts[i], file=f_topK)
        

    f_top1.close()
    f_topK.close()

def datasetBuilding(config):
    LOG.log('Building Dataset Setting')
    settingPath = "settings/dataset/newData.json"
    data = {
        "name":"newData",
        "method":"build",
        "max_input_len": 100,
        "max_output_len": 50,
        "match": True,
        "Parts":
        {
            "train":
            {
                "name":"train",
                "path":config.train_prefix,
                "sorted": True,
                "shuffled": False,
            },
            "valid":
            {
                "name":"valid",
                "path":config.valid_prefix,
                "sorted": True,
                "shuffled": False
            }
        }
    }
    saveToJson(settingPath, data)
    return settingPath

def takeEmbedding():
    net = BertForMaskedLM.from_pretrained('bert-base-uncased')
    embedding = net.bert.embeddings.word_embeddings
    return embedding


def argLoader():
    parser = argparse.ArgumentParser()

    # Options Setting
    parser.add_argument('--dataset', type=str, default='gigaword')
    parser.add_argument('--data_part', type=str, default='test')

    # Device Setting
    parser.add_argument('--dataLoader_workers', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)

    # Model Saving Setting
    parser.add_argument('--save_path', type=str, default='./model')
    # Actions
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")
    parser.add_argument('--do_test', action='store_true', help="Whether to run test")

    # Path for Input
    parser.add_argument('--inputFile', type=str, default='none')
    parser.add_argument('--train_prefix', type=str, default='train')
    parser.add_argument('--valid_prefix', type=str, default='valid')

    # Learning Parameters
    parser.add_argument('--prob_src', type=float, default=0.1)
    parser.add_argument('--prob_tgt', type=float, default=0.9)
    parser.add_argument('--lRate', type=float, default=4e-5)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=8)

    # Network Parameters
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_vocab', type=int, default=30522)

    parser.add_argument('--max_len', type=int, default=5000)
    parser.add_argument('--n_token_type', type=int, default=2)
    parser.add_argument('--hidden_act', type=str, default='relu')
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--diff_src', action='store_true', help="Whether to use different encoder hiddens")
    
    # Training  parameters
    parser.add_argument('--loss_mode', type=int, default=0)
    parser.add_argument('--dataParallel', action='store_true', help="Whether dataParallel or not.")

    # Which Portion to Use
    parser.add_argument('--seen_prob', type=float, default=1.0)
    parser.add_argument('--similar_prob', type=float, default=1.0)
    parser.add_argument('--unseen_prob', type=float, default=1.0)
    parser.add_argument('--similar_threshold', type=float, default=1.0)

    # Pretrained model
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--PAD', type=int, default=0)
    parser.add_argument('--UNK', type=int, default=100)
    parser.add_argument('--CLS', type=int, default=101)
    parser.add_argument('--SEP', type=int, default=102)
    parser.add_argument('--MASK', type=int, default=103)

    # Testing Parameters
    parser.add_argument('--test_model', type=str, default='./model/model_best.pth.tar')
    parser.add_argument('--search_method', type=str, default='BFS_BEAM')
    parser.add_argument('--reranking_method', type=str, default='none')

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--cands_limit', type=int, default=100000)
    parser.add_argument('--answer_size', type=int, default=5)
    parser.add_argument('--gen_max_len', type=int, default=50)
    parser.add_argument('--gamma_value', type=float, default=14.0)
    parser.add_argument('--beta_value', type=float, default=0.5)
    parser.add_argument('--reward', type=float, default=0.25)
    parser.add_argument('--no_biGramTrick', action='store_true', help='Wheter do not biGramTrick')
    parser.add_argument('--no_triGramTrick', action='store_true', help='Wheter do not triGramTrick')

    args = parser.parse_args()
    args.do_eval = True
    args.biGramTrick = not args.no_biGramTrick
    args.triGramTrick = not args.no_triGramTrick

    embeddingFunc = takeEmbedding()

    args.para = {
        'seen_prob': args.seen_prob,
        'similar_prob': args.similar_prob,
        'unseen_prob': args.unseen_prob,
        'similar_threshold': args.similar_threshold,
        'embedding': embeddingFunc
    }

    args.suffix = '_' + args.dataset\
                  + '_' + args.data_part\
                  + '_' + args.search_method\
                  + '_' + str(args.beam_size)\
                  + '_' + str(args.answer_size)\
                  + '_' + str(args.gamma_value)\
                  + '_' + str(args.biGramTrick)\
                  + '_' + str(args.triGramTrick)\
                  + '_' + str(args.reranking_method)
    
    if args.reranking_method == "bounded_word_reward":
        args.suffix += '_' + str(args.reward)

    if args.do_test:
        if (args.inputFile == 'none'):
            print('No testing input file. Please use "--inputFile example.txt".')
            return args
        args.optionFrames = {
            'test': 'settings/test/test.json'
        }

    elif args.do_train:
        if (not path.exists(args.train_prefix+'.Ndocument')) or (not path.exists(args.train_prefix + '.Nsummary')):
            print('No training input file. Please use "--train_prefix train" to assign "train.Ndocument" and "train.Nsummary"')
            return args

        if (not path.exists(args.valid_prefix+'.Ndocument')) or (not path.exists(args.valid_prefix + '.Nsummary')):
            print('No validation input file. Please use "--valid_prefix valid" to assign "valid.Ndocument" and "valid.Nsummary"')
            return args

        args.optionFrames = {}
        args.optionFrames['dataset'] = datasetBuilding(args)
        args.optionFrames['training'] = "settings/training/gigaword_" + str(args.batch_size) + ".json"

    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    args = argLoader()

    if args.device:
        torch.cuda.set_device(args.device)

    print('CUDA', torch.cuda.current_device())

    if args.do_train:
        train(args)

    elif args.do_test:
        test(args)

if __name__ == '__main__':
    main()

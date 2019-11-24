import torch.utils.data as data

from utility import *


class myDataSet(data.Dataset):
    def __init__(self, name, batch_size, lenFunc, prepareFunc, options, log, mode='train'):
        self.name = name
        self.log = log
        self.batch_size = batch_size
        self.len_func = lenFunc
        self.prepare_func = prepareFunc
        self.mode = mode

        self.path = options['Parts'][name]['path']
        self.sorted = options['Parts'][name]['sorted']
        self.shuffled = options['Parts'][name]['shuffled']

        self.source_len = options['max_input_len']
        self.target_len = options['max_output_len']
        self.match = options['match']

        self.n_data = 0
        self.Data = []
        self.n_batch = 0
        self.Batch = []
        self.config = None

    def setConfig(self, config, prob_src, prob_tgt):
        self.config = cp(config)
        self.config.prob_src = prob_src
        self.config.prob_tgt = prob_tgt

    def sortByLength(self):
        self.log.log('Start sorting by length')
        data = self.Data
        number = self.n_data

        lengths = [(self.len_func(data[Index]), Index) for Index in range(number)]
        sorted_lengths = sorted(lengths)
        sorted_Index = [d[1] for d in sorted_lengths]

        data_new = [data[sorted_Index[Index]] for Index in range(number)]

        self.Data = data_new
        self.log.log('Finish sorting by length')

    def shuffle(self):
        self.log.log('Start Shuffling')

        data = self.Data
        number = self.n_data

        shuffle_Index = list(range(number))
        random.shuffle(shuffle_Index)

        data_new = [data[shuffle_Index[Index]] for Index in range(number)]

        self.Data = data_new
        self.log.log('Finish Shuffling')

    def genBatches(self):
        batch_size = self.batch_size
        data = self.Data
        number = self.n_data
        n_dim = len(data[0])

        number_batch = number // batch_size
        batches = []

        for bid in range(number_batch):
            batch_i = []
            for j in range(n_dim):
                data_j = [item[j] for item in data[bid * batch_size: (bid + 1) * batch_size]]
                batch_i.append(data_j)
            batches.append(batch_i)

        if (number_batch * batch_size < number):
            batch_i = []
            for j in range(n_dim):
                data_j = [item[j] for item in data[number_batch * batch_size:]]
                batch_i.append(data_j)
            batches.append(batch_i)
            number_batch += 1

        self.n_batch = number_batch
        self.Batch = batches

    def load(self):
        pass

    def afterLoad(self):
        if self.sorted:
            self.sortByLength()
        if self.shuffled:
            self.shuffle()

        # Generate Batches
        self.log.log('Generating Batches')
        self.genBatches()

    def __len__(self):
        if self.mode == 'train' or self.mode == 'valid':
            return self.n_batch
        return self.n_data

    def __getitem__(self, index):
        if self.mode == 'train':
            source, target = self.Batch[index]
            return self.prepare_func(source, target, self.config, self.config.prob_src, self.config.prob_tgt,
                                     self.config.para)
        if self.mode == 'valid':
            source, target = self.Batch[index]
            return self.prepare_func(source, target, self.config, 0.0, self.config.prob_tgt, self.config.para)
        source, target = self.Data[index]
        return prepare_test_data(source, target, self.config)


class myDataSet_Vocab(myDataSet):
    def __init__(self, name, batch_size, lenFunc, prepareFunc, Vocab, options, log, mode='train'):
        super(myDataSet_Vocab, self).__init__(name, batch_size, lenFunc, prepareFunc, options, log, mode)
        self.Vocab = Vocab
        # Loading Dataset
        self.log.log('Building dataset %s from orignial text documents' % (self.name))
        self.n_data, self.Data = self.load()
        self.log.log('Finish Loading dataset %s' % (self.name))

        self.afterLoad()

    def load(self):
        srcFile = open(self.path + '.Ndocument', 'r', encoding='utf-8')
        refFile = open(self.path + '.Nsummary', 'r', encoding='utf-8')
        data = []

        Index = 0
        while True:
            Index += 1

            srcLine = srcFile.readline()
            if (not srcLine):
                break
            refLine = refFile.readline()
            if (not refLine):
                break

            srcLine = remove_digits(srcLine.strip()).lower()
            refLine = remove_digits(refLine.strip()).lower()

            # Processing Input Sequences
            if (self.match) and ('train' in self.name):
                match = len(set(srcLine.split()) & set(refLine.split()))
                if match < 3:
                    continue

            if (len(srcLine.split()) < 1) or (len(refLine.split()) < 1):
                continue

            document = Sentence2ListOfIndex(srcLine, self.Vocab)
            summary = Sentence2ListOfIndex(refLine, self.Vocab)

            if len(document) > self.source_len:
                document = document[:self.source_len]

            if len(summary) > self.target_len:
                summary = summary[:self.target_len]

            document = document
            summary = summary

            data.append([document, summary])

        return len(data), data


class myDataSet_Bert(myDataSet):
    def __init__(self, name, batch_size, lenFunc, prepareFunc, Tokenizer, options, log, mode='train'):
        super(myDataSet_Bert, self).__init__(name, batch_size, lenFunc, prepareFunc, options, log, mode)
        self.Tokenizer = Tokenizer
        # Loading Dataset
        self.log.log('Building dataset %s from orignial text documents' % (self.name))
        self.n_data, self.Data = self.load()
        self.log.log('Finish Loading dataset %s' % (self.name))

        self.afterLoad()

    def load(self):
        srcFile = open(self.path + '.Ndocument', 'r', encoding='utf-8')
        refFile = open(self.path + '.Nsummary', 'r', encoding='utf-8')
        data = []

        Index = 0
        while True:
            Index += 1
            srcLine = srcFile.readline()
            if (not srcLine):
                break
            refLine = refFile.readline()
            if (not refLine):
                break
            srcLine = srcLine.strip()
            refLine = refLine.strip()
            src_tokens = srcLine.split()
            ref_tokens = refLine.split()

            if (len(src_tokens) < 1) or (len(ref_tokens) < 1):
                continue

            if (self.match) and ('train' in self.name):
                match = len(set(src_tokens) & set(ref_tokens))
                if match < 3:
                    continue

            if len(src_tokens) > self.source_len:
                src_tokens = src_tokens[:self.source_len]
                srcLine = " ".join(src_tokens)

            if len(ref_tokens) > self.target_len:
                ref_tokens = ref_tokens[:self.target_len]
                refLine = " ".join(ref_tokens)

            document = self.Tokenizer.convert_tokens_to_ids(self.Tokenizer.tokenize(srcLine))
            summary = self.Tokenizer.convert_tokens_to_ids(self.Tokenizer.tokenize(refLine))

            if len(document) + len(summary) > 200:
                continue

            document = document
            summary = summary

            data.append([document, summary])

        return len(data), data

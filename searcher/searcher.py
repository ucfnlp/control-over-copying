from utility import *


class Searcher:
    def __init__(self, net, config):
        self.net = net
        self.config = config
        self.search_method = eval("self." + config.search_method)

    def BFS_BEAM(self, source, target=[]):
        score = []
        reward = []

        startState = [score, target, reward]
        Candidates = [[startState]]
        Answers = []

        for i in range(self.config.gen_max_len + 1):
            Cands = Candidates[i]
            topK = []
            for cand in Cands:
                score, target, reward = cand
                inputs, positions, token_types, masks = prepare_test_data(source, target, self.config)

                with torch.no_grad():
                    predicts = self.net(inputs, positions, token_types, masks)
                    predicts = predicts[0, -1]
                    predicts = do_tricks(predicts, source, target, self.config)
                    predicts_topK = torch.topk(predicts, self.config.beam_size)
                    new_cands = predicts_topK[1]
                    reward_i = float(- predicts_topK[0].sum() / self.config.beam_size)
                    scores = predicts[new_cands]

                    for new_cand, new_score in zip(new_cands.tolist(), scores.tolist()):
                        new_scores = cp(score) + [- float(new_score)]
                        new_target = cp(target) + [new_cand]
                        new_reward = cp(reward) + [reward_i]
                        topK.append([new_scores, new_target, new_reward])

            topK = sorted(topK, key=lambda x: sum(x[0]))[:self.config.beam_size * 2]
            topK_next = []
            for score, seq, reward in topK:
                if seq[-1] == self.config.SEP:
                    Answers.append([score, seq, reward])
                else:
                    topK_next.append([score, seq, reward])

            if (len(Answers) >= self.config.answer_size):
                break

            if (len(topK_next) > self.config.beam_size):
                topK_next = topK_next[:self.config.beam_size]

            Candidates.append(topK_next)

        if (len(Answers) < 1):
            return sorted(Candidates[-1], key=lambda x: sum(x[0]))

        Answers = sorted(Answers, key=lambda x: sum(x[0]))
        # Answers = sorted(Answers, key=lambda x: sum(x[0])/(len(x[1]) - 1 + 1e-8))
        if (len(Answers) > self.config.answer_size):
            Answers = Answers[:self.config.answer_size]

        return Answers

    def Greedy(self, source, target=[]):
        score = []
        startState = [score, target]
        Candidates = [startState]

        for i in range(self.config.gen_max_len + 1):
            score, target = Candidates[i]
            if (len(target) > 0) and (target[-1] == self.config.SEP):
                break
            inputs, positions, token_types, masks = prepare_test_data(source, target, self.config)
            with torch.no_grad():
                predicts = self.net(inputs, positions, token_types, masks)
                predicts = predicts[0, -1]
                predicts = do_tricks(predicts, source, target, self.config)
                new_cand = int(torch.argmax(predicts))
                new_score = cp(score) + [- float(predicts[new_cand])]
                new_target = cp(target) + [new_cand]
                Candidates.append([new_score, new_target])

        return [Candidates[-1]]

    def length_Predict(self, source, target=[]):
        score, target = self.Greedy(source, target)[0]
        return len(target)

    def search(self, source):
        return self.search_method(source)

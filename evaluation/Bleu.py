from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu as bleu


class Bleu(object):
    def __init__(self, settings):
        self.settings = settings

    def eval(self, hypList, refList):
        # Lower
        hypList = [it.lower() for it in hypList]
        refList = [it.lower() for it in refList]
        number = len(hypList)
        n_ref = len(refList) // number

        result = {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'bleu': 0.0
        }

        for Index in range(0, number):
            ref = [refList[i].split() for i in range(Index * n_ref, (Index + 1) * n_ref)]
            ref = [r[:-1] if r[-1] == '.' else r for r in ref]
            hyp = hypList[Index].split()
            if (hyp[-1] == '.'):
                hyp = hyp[:-1]

            Smooth = SmoothingFunction()

            bleu_1 = bleu(ref, hyp, weights=[1], smoothing_function=Smooth.method1)
            bleu_2 = bleu(ref, hyp, weights=[0, 1], smoothing_function=Smooth.method1)
            bleu_3 = bleu(ref, hyp, weights=[0, 0, 1], smoothing_function=Smooth.method1)
            bleu_4 = bleu(ref, hyp, weights=[0, 0, 0, 1], smoothing_function=Smooth.method1)
            bleu_all = bleu(ref, hyp, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=Smooth.method1)

            result['bleu_1'] += bleu_1
            result['bleu_2'] += bleu_2
            result['bleu_3'] += bleu_3
            result['bleu_4'] += bleu_4
            result['bleu'] += bleu_all

        result['bleu_1'] /= number
        result['bleu_2'] /= number
        result['bleu_3'] /= number
        result['bleu_4'] /= number
        result['bleu'] /= number

        return result

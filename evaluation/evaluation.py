class evalFile(object):
    def __init__(self, hyp_fileName, ref_fileName, metrics):
        self.hyp_fileName = hyp_fileName
        self.ref_fileName = ref_fileName
        self.metrics = metrics

    def eval(self):
        hypFile = open(self.hyp_fileName, 'r')
        refFile = open(self.ref_fileName, 'r')

        hypList = hypFile.readlines()
        refList = refFile.readlines()

        result = {}
        for metricName, metricSetting in self.metrics.items():
            Obj = eval(metricName)(metricSetting)
            result[metricName] = Obj.eval(hypList, refList)

        return result


class evalList(object):
    def __init__(self, metrics):
        self.metrics = metrics

    def eval(self, hypList, refList):
        result = {}
        for metricName, metricSetting in self.metrics.items():
            Obj = eval(metricName)(metricSetting)
            result[metricName] = Obj.eval(hypList, refList)
        return result


def evaluate(hyp_fileName, ref_fileName, metrics, log, Show=True):
    Eval = evalFile(hyp_fileName, ref_fileName, metrics)
    result = Eval.eval()
    if Show:
        for kk, vv in result.items():
            print(kk)
            print(vv)
    return result

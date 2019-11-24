from collections import OrderedDict

from utility import *


def dispDict(x, prefix=""):
    s = ""
    if (type(x) == dict) or (type(x) == OrderedDict):
        s += "\n" + prefix + "{"
        for k, v in x.items():
            s += "\n\t" + prefix + k + " : " + dispDict(v, prefix + "\t")
        s += "\n" + prefix + "}"
    else:
        s += str(x)
    return s


def optionsLoader(log, optionFrames, disp=False, reload=None):
    if (reload == None):
        log.log('Loading Options Frames')
        options = OrderedDict()
        for k, v in optionFrames.items():
            log.log("%s options are loading " % (k))
            option = loadFromJson(v)
            options[k] = option
    else:
        log.log('Reloading Options')
        options = loadFromJson(reload)

    if disp:
        log.log("\nOptions:\n" + dispDict(options))
    return options

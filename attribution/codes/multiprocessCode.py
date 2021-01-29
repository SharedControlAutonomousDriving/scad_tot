import numpy as np
from lime import lime_tabular

def multiprocessCode(args):
    e,sample,f = args
    content = e(sample, f, labels=[0,1,2,3,4]).as_map()
    lime_values = np.zeros((5, 25))
    for c in list(content.keys()):
        samp_value = np.zeros(25)
        for j, k in np.array(content[c]):
            samp_value[int(j)] = k
        lime_values[c, :] = samp_value

    return lime_values


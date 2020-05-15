import numpy as np
from tot_net import TOTNet
from tensorflow.keras.models import load_model

def find_local_robustness_boundaries(model_path, x, samples, d_min=0.0001, d_max=100.00):
    def find_distance(sample, sign):
        pass
    model = load_model(model_path)
    results = [(-1*find_distance(s, -1), find_distance(s, +1)) for s in samples]
    ld = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    rd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    return ((ld, rd), results)



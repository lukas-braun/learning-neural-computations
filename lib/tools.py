import numpy as np
import itertools

# Mean squared error
def mse(data, target):
    axis = 1 if data.ndim == 2 else 0
    return np.mean((data - target[np.newaxis, :])**2, axis=axis)


# Signed relative error
def dist(data, target):
    if data.ndim == 2:
        axis = 1
        n = data.shape[1]
    else:
        axis = 0
        n = 1
        if isinstance(target, (float, int)):
            target = np.asarray([target])
    norm = np.max([np.linalg.norm(target), 0.075])
    
    return np.sum((data - target[np.newaxis, :]) / norm, axis=axis)


# Low-pass filter
def lpf(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


# Apply convergence criterion
def converged(datas, targets, tolerances):
    convs = []
    for i, (data, target) in enumerate(zip(datas, targets)):
        convs.append(np.abs(dist(data, target)) < tolerances[i])
    return np.sum(convs, axis=0) / len(targets)


# Return possible parameter combinations
def combinations(level, as_str=False):
    comb = "{" if as_str else []
    for r in range(1, level+1):
        for s in itertools.combinations(range(level), r):
            levels = [False] * level
            for pos in s:
                levels[pos] = True
            if as_str:
                comb += "\n" + str(levels).replace("[", "{").replace("]", "}").lower() + ","
            else:
                comb.append(levels)
    if as_str:
        comb += "\n};"
    return comb


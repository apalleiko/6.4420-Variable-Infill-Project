import numpy as np
import scipy as sp
import scipy.io

mat_contents = sp.io.loadmat('../layers.mat')
layers = []
for idx,layer in enumerate(mat_contents['layers'][0,:]):
    layers.append({'X':[float(i) for i in layer[0]],
                   'Y':[float(i) for i in layer[1]],
                   'Z':[float(i) for i in layer[2]],
                   'stresses':[float(i) for i in layer[3]]})



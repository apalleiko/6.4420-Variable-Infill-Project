import numpy as np
import scipy as sp
import scipy.io


def mat2dict(path):
    mat_contents = sp.io.loadmat(path)
    layers = []
    for idx,layer in enumerate(mat_contents['layers'][0,:]):
        layers.append({'X':[float(i) for i in layer[0]],
                       'Y':[float(i) for i in layer[1]],
                       'Z':[float(i) for i in layer[2]],
                       'stresses':[float(i) for i in layer[3]]})

    return layers


if __name__ == "__main__":
    path = "/cube_layers.mat"
    output = mat2dict(path)



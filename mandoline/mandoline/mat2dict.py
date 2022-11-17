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


# def center(self, cp):
#     """Centers the model at the given centerpoint cp."""
#     cx = (self.points.minx + self.points.maxx)/2.0
#     cy = (self.points.miny + self.points.maxy)/2.0
#     cz = (self.points.minz + self.points.maxz)/2.0
#     self.translate((cp[0]-cx, cp[1]-cy, cp[2]-cz))
#
# model.center( (self.center_point[0], self.center_point[1], (model.points.maxz-model.points.minz)/2.0) )

if __name__ == "__main__":
    path = "/cube_layers.mat"
    output = mat2dict(path)



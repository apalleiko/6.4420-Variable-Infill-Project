import scipy as sp
import scipy.io
import numpy as np


class fea(object):
    def __init__(self, fea_path):
        self.fea_path = fea_path
        self.fea_map = []
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        self.stresses = np.array([])
        self.normalized_stresses = np.array([])
        self.mat2dict()
        self.normalize_stresses()

    def mat2dict(self):
        mat_contents = sp.io.loadmat(self.fea_path)
        layers = []
        for idx, layer in enumerate(mat_contents['layers'][0, :]):
            layers.append(
                {
                    'coords': np.array(
                        [[float(layer[0][idx]),
                          float(layer[1][idx]),
                          float(layer[2][idx])] for idx in range(len(layer[0]))]
                    ),
                    'stresses': np.array([float(i) for i in layer[3]])})
            cur_layer_stresses = layers[idx]['stresses']
            self.stresses = np.concatenate((self.stresses,cur_layer_stresses))
            cur_layer_coords = layers[idx]['coords']
            self.xs = np.concatenate((self.xs,cur_layer_coords[:,0]))
            self.ys = np.concatenate((self.ys, cur_layer_coords[:, 1]))
            self.zs = np.concatenate((self.zs, cur_layer_coords[:, 2]))
        self.fea_map = layers

    def regenerate_coords(self):
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        for idx, layer in enumerate(self.fea_map):
            cur_layer_stresses = self.fea_map[idx]['stresses']
            self.stresses = np.concatenate((self.stresses, cur_layer_stresses))
            cur_layer_coords = self.fea_map[idx]['coords']
            self.xs = np.concatenate((self.xs, cur_layer_coords[:, 0]))
            self.ys = np.concatenate((self.ys, cur_layer_coords[:, 1]))
            self.zs = np.concatenate((self.zs, cur_layer_coords[:, 2]))

    def translate(self, offset):
        """Translates the coordinates of this point."""
        offset = np.array([offset])
        for idx,layer in enumerate(self.fea_map):
            self.fea_map[idx]['coords'] += offset

    def center_with_slicer(self,center_point):
        """Centers the model at the given centerpoint cp."""
        cx = (min(self.xs) + max(self.xs))/2.0
        cy = (min(self.ys) + max(self.ys))/2.0
        cz = (min(self.zs) + max(self.zs))/2.0
        self.translate((center_point[0]-cx, center_point[1]-cy, center_point[2]-cz))
        # print('before',self.xs)
        self.regenerate_coords()
        # print('after',self.xs)

    def normalize_stresses(self):
        self.normalized_stresses = np.array(self.stresses-min(self.stresses))/(max(self.stresses)-min(self.stresses))

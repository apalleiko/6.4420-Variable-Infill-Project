import scipy as sp
import scipy.io
import numpy as np
import matlab.engine


class fea(object):
    def __init__(self, stl_path,fea_active,layer_height):
        self.stl_path = stl_path
        self.fea_active = fea_active
        self.layer_height = layer_height
        if self.fea_active != 'default':
            self.runFEA()
        self.fea_map = []
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        self.stresses = np.array([])
        self.normalized_stresses = np.array([])
        self.mat2dict()
        self.normalize_stresses()

    def runFEA(self):
        assert self.fea_active == 'custom', "to use fea, enter keyword custom or default after --fea"
        satisfied = 'N'
        while satisfied != 'Y':
            eng = matlab.engine.start_matlab()
            smodel = eng.FEA_run_model(self.stl_path,nargout=1)
            empty = '[]'
            fixedVertices = str(input("List vertices to fix as Python array (for two fixed vertices at V1 and V2, input --> ex. [1,2]): ") or empty)
            fixedFaces = str(input("List faces to fix as Python array (for two fixed faces at F1 and F3, input --> ex. [1,3]): ") or empty)
            loadedFaces = str(input("List faces to load as Python array (for two loaded faces at F1 and F3, input --> ex. [1,3]): ") or empty)
            faceForces = str(input("List 3D force vectors (N) to apply to faces as a Python array of arrays (for two loaded faces, input --> ex. [[-10; 5; 0],[0; -10; 0]]): ") or empty)
            loadedVertices = str(input("List vertices to load as Python array (for two loaded vertices at V1 and V2, input --> ex. [1,2]): ") or empty)
            vertexForces = str(input("List 3D force vectors (N) to apply to vertices as a Python array of arrays (with two loaded vertices, input --> ex. [[-10; 5; 0],[0; -10; 0]]): ") or empty)
            eng.FEA_solve(smodel, self.stl_path,self.layer_height,fixedVertices,fixedFaces,loadedFaces,faceForces,loadedVertices,vertexForces,nargout=0)
            satisfied = input("\n[Y] to proceed with slicing, [N] to repeat FEA: ")

    def mat2dict(self):
        if self.fea_active == 'default':
            mat_contents = sp.io.loadmat('../fea_output_default.mat')
        else:
            mat_contents = sp.io.loadmat('../fea_output.mat')
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

    def get_normalized_layer_stresses(self,idx):
        cur_stresses = self.fea_map[idx]['stresses']
        return np.array(cur_stresses-min(self.stresses))/(max(self.stresses)-min(self.stresses))

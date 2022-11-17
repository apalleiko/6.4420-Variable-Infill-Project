import numpy as np


import mandoline.util as mshu
import logging

logger = logging.getLogger(__name__)

from mandoline.util import fix, simpvol2D, connect_elm, connect_elm2ed
# import msh.plot as plt
import copy


# import vtk

class Mesh(object):
    """
    Parent class of Mesh2D and Mesh3D; contains shared functionality.
    Aim is to prevent duplication of code common to both Mesh2D and Mesh3D
    classes, since they otherwise would not share a parent.
    """

    def elmtype_connectivity(self, elmtype):
        """ returns the portion of the connectivity array for a particular
        element type (with padding -1s trimmed!)
        :param elmtype: integer element type as in self.elm_type
        """
        matching_elms = (self.elm_type == elmtype)
        # # CF, Aug 14, 2019:
        # extra [:] index needed for 3D extrude meshes, if not cast to an np
        # array through the implicit copy done with [:], won't index
        # correctly...
        elmtype_connectivity = self.elm[:][matching_elms, ...]

        # find first index where -1 padding begins; index up until that point
        padding_begins = np.sum(elmtype_connectivity[0, ...] >= 0)
        return elmtype_connectivity[:, :padding_begins]

    def elmtype_vertices(self, elmtype):
        """ returns the vertex locations for a mesh element type
        :param elmtype: integer element type as in self.elm_type
        :returns: np.ndarray shape (n_elements, n vert per element, dim)
        containing the locations of the vertices for the elements of type elmtype
        .. note:: the reason this is a method is because the vert array is
        implicitly 3D and we have to trim the output to be the correct dimension
        for the mesh.
        """
        return self.vert[self.elmtype_connectivity(elmtype)][..., :self.dim]

    def vertices_on_edge(self, global_edge):
        """ returns the global vertex numbers given a global edge number of the
        mesh
        :param global_edge: int or array of global edge numbers
        Recall that ed2ed has the following structure for each row
        (corresponding to a single edge in the mesh:
        [(+) elm #, (+) side local edge #, (-) elm#, (-) elm loc edge #, vertex
        1, vertex 2...] (0:2) (2:4) (4:) """
        return self.ed2ed[global_edge, 4:]

    def edgetype_vertex_numbers(self, edge_type):
        """ returns the vertices making up each edge for a given edge type
        :param edge_type: integer edge type
        :returns: np.ndarray shape (global edge number, n nvertices on edge)
        """
        matching_edge_indices = (self.ed_type == edge_type)
        # potentially padded with negative numbers past a certain index
        edgetype_vertices = self.ed2ed[matching_edge_indices, 4:]
        padding_begins = np.sum(edgetype_vertices[0, ...] >= 0)
        return edgetype_vertices[:, :padding_begins]

    def edgetype_vertices(self, edge_type):
        """ returns the vertex locations for a mesh edge type
        :param edge_type: integer edge type as defined in self.ed_type
        :returns: np.ndarray of shape (n typed edges, vertex, coordinate)
        """
        edgetype_vertices = self.edgetype_vertex_numbers(edge_type)

        # CF: 2019-08-18
        # extra [:] index needed for 3D extrude meshes, if not cast to an np
        # array through the implicit copy done with [:], won't index
        # correctly...
        vertices = self.vert[:][edgetype_vertices, :self.dim]
        return vertices

    def global2typed_elm(self, global_elm):
        """
        Returns tuple (elm type, typed elm num) from global elm num
        :param global_elm: integer global element number
        :returns: tuple (element type, typed element num)
        """
        elm_type = self.elm_type[global_elm]
        typed_elm = self.glob2type_elm[global_elm]
        return elm_type, typed_elm

    def global2typed_edge(self, global_edge):
        """
        Returns tuple (ed type, typed ed num) from global ed num
        :param global_ed: integer global edge number
        :returns: tuple (ed type, typed ed num)
        """
        return self.ed_type[global_edge], self.glob2type_ed[global_edge]

    def _typed2global_edge(self, edge_type, typed_edge):
        """
        Returns int global edge from edge type and typed edge number
        :param ed_type: int edge type
        :param typed_ed_num: int typed edge number
        :returns: int global edge number
        :warning: expensive to compute, for debugging only
        """
        glob_idx_match = np.where(self.glob2type_ed == typed_ed)[0]
        which_match = np.where(self.ed_type[glob_idx_match] == ed_type)[0]
        glob_idx = glob_idx_match[which_match]
        assert len(glob_idx) == 1
        return glob_idx[0]

    def typed2exterior_edge(self, edge_type, typed_edge):
        """
        Returns int typed exterior edge number from (edType, typed_ed)
        :param ed_type: int edge type
        :param typed_ed_num: int or int arr of typed edge number(s)
        :returns: int typed exterior edge number or int array of typed exterior
            edge numbers. If an (edtype, typed_ed) pair is an interior edge, the
            return value will be -1.
        """
        return self.in2ex_bd_id[edge_type][typed_edge]

    def is_exterior_edge(self, ed_type, typed_edge):
        """
        Returns whether the ed_type, typed_ed_num is an exterior ed
        :param ed_type: int edge type
        :param typed_ed_num: int or int arr of typed edge number(s)
        :returns: bool if (edType, typed_ed_num) pair is an exterior edge or
            bool array if typed_ed_num was an array.
        """
        return self.ids_exterior_ed_by_type[ed_type][typed_edge]

    def elm_adjacent_to_edge(self, global_edge, side):
        """
        Returns the adjacent element number on the specified side of the
        given global edge number
        :param global_edge: integer or array of global edge numbers
        :param side: 'LEFT' or 'RIGHT'
        NOTE:in ed2ed the LEFT element is given first, then the right element.
        If boundary edge, adjacent RIGHT element is -1
        """
        # assign the index based on the structure of the ed2ed rows
        if side == 'LEFT':
            elmIdx = 0
        elif side == 'RIGHT':
            elmIdx = 2
        else:
            raise ValueError("Invalid edge side specification")
        return self.ed2ed[global_edge, elmIdx]

    def global_ed2elm_local_edge(self, global_edge, side):
        """
        For a global edge, returns the corresponding local edge number for the
        adjacent element on the specified side.
        :param global_edge: int or array of global edge numbers
        :param side: either 'LEFT' or 'RIGHT'
        Each global edge has a left element, and possibly a right element (not
        in the case of a boundary). In the case where the adjacent element
        exists, the global edge is spatially corresponds to one of the local
        edges on the adjacent element. This information is contained in the
        ed2ed connectivity array, and this helper function represents a way for
        the user of the mesh class to retrieve this information without knowing
        the internals of the mesh data structure.
        NOTE: if the element to the right hand side does not exist (boundary
        case), then the return value will be -1.
        """
        # assign the index based on the structure of the ed2ed rows
        if side == 'LEFT':
            elmIdx = 0
        elif side == 'RIGHT':
            elmIdx = 2
        else:
            raise ValueError("Invalid edge side specification")
        return self.ed2ed[global_edge, elmIdx + 1]

    def exterior2typed_edge(self, edge_type, exterior_edge):
        """
        Returns typed edge number from edge_type and exterior edge number
        :param edge_type: int edge type
        :param exterior_edge: int typed exterior edge number
        NOTE: external edge numbers are typed
        """
        return np.where(self.in2ex_bd_id[edge_type] == exterior_edge)[0][0]

    def typed2exterior_edge(self, edge_type, typed_edge):
        """
        returns the exterior edge ID from an (edgeType, edge) pairing
        :param edge_type: int edge type
        :param typed_edge: int typed edge number
        """
        return self.in2ex_bd_id[edge_type][typed_edge]

    def build_boundary_elm_sets(self):
        """
        Computes a list of sets that hold typed_elm numbers on the boundary
        """
        globally_numbered_bd_elms = set(self.ed2ed[self.ed2ed[:, 2] < 0, 0])
        bd_elms = [set() for eType in self.n_elm_type]
        for elm in globally_numbered_bd_elms:
            elmType, typed_elm = self.elm_type[elm], self.glob2type_elm[elm]
            bd_elms[elmType].add(typed_elm)
        return bd_elms

    def is_boundary_elm(self, elm_type, typed_elm):
        """
        Returns whether the elm specified by elm_type, typed_elm_num is a
        boundary element.
        :param elm_type: int element type
        :param typed_elm_num: int or int arr of typed elm number(s)
        :returns: bool if (edType, typed_ed_num) pair is an exterior edge or
            bool array if typed_ed_num was an array.
        """
        return typed_elm in self.boundary_elms[elm_type]

    def build_type2glob_elm_list(self):
        """
        builds the element by type to global element lookup arrays
        type2glob_elm[elmType][elm] = global elm #
        """
        type2glob = [-np.ones(nElm, dtype=int) for nElm in self.n_elm_type]
        for global_elm, elm_type in enumerate(self.elm_type):
            typed_elm = self.glob2type_elm[global_elm]
            type2glob[elm_type][typed_elm] = global_elm
        assert all([(arr >= 0).all() for arr in type2glob])  # none unassigned
        return type2glob

    def typed_edges_on_element(self, elm_type, typed_elm):
        """
        Returns the typed edge numbers around the element
        """
        global_elm = self.type2glob_elm[elm_type][typed_elm]
        edges = self.elm2ed[global_elm, :]
        edges = edges[edges >= 0]  # remove padding -1s
        return [self.global2typed_edge(ed) for ed in edges]

    def bcid_global_edges(self, bcid):
        """ returns the global edge numbers of edges matching a bcid number
        :param id_label: The bcid integer as determined by the index in
            the list of lambda functions passed in when self.set_bc_ids is
            called.
        :returns: integer global edge numbers of the edges with BCID number bcid
        The array self.ed2ed denotes bcid labels by storing the integer
        (-bcid -1) in column 2, which is what we use to do the lookup, this
        number is different from bcid in order to differentiate the number from
        the positive edge numbers.
        """
        lookup_integer = -bcid - 1
        return np.where(self.ed2ed[:, 2] == lookup_integer)[0]

    def bcid_global_edges_from_label(self, label):
        """ returns the global edge numbers of edges matching a given label
        :param label: string label provided by user in self.bcid_labels
        :returns: integer global edge numbers of the edges with BCID number bcid
        This is a convenience method to reference bcid edge numbers according to
        the user specified labels passed in when self.set_bc_ids is called. The
        function merely looks up the bcid associated with the label and calls
        self.bcid_global_edges.
        """
        bcid_corresponding_to_label = self.bcid_labels[label]
        return self.bcid_global_edges(bcid_corresponding_to_label)

    def vertex_orientation_num(self, v1, v2):
        """ returns enumeration number of the relative orientations between two
        set of global vertices, v1 and v2
        :param v1: global vertex numbers of the 'static' face or edge
        :param v2: global vertex numbers of the 'orientted' face or edge
        :returns: integer orientation between the two vertex sets
        The two vertex sets are oriented relative to each other; this
        orientation is enumerated for all possible vertex configurations of
        triangular and rectangular planar edges with vertices in CCW order. The
        enuemration can be used to find the permutation of the edge/face DOF
        on v2 relative to v1 via master.edge_vertex_numbers[enum]. That is, the
        orientation enumeration specifies the permutation of DOF to align the
        element face basis functions with the edge basis functions, or
        vice-versa.
        2D meshes
        Edge is defined by 2 points, [a, b]
            Other edge: [a, b], orient number [0, 0], 0 if one_num
            Other edge: [b, a], orient number [0, 1], 1 if one_num
        3D Meshes
        Triangle: Edge is defined by 3 points, [a, b, c]
           Other edge [a, b, ...], orient number [0, 0], 0 if one_num
           Other edge [b, c, ...], orient number [1, 0], 1 if one_num
           Other edge [c, a, ...], orient number [2, 0], 2 if one_num
           Other edge [a, c, ...], orient number [1, 0], 4 if one_num
           Other edge [b, a, ...], orient number [1, 1], 5 if one_num
           Other edge [c, b, ...], orient number [1, 2], 6 if one_num
        Rectangle: Edge is defined by 4 points, [a, b, c, d]
           Other edge [a, b, ...], orient number [0, 0], 0 if one_num
           Other edge [b, c, ...], orient number [1, 0], 1 if one_num
           Other edge [c, d, ...], orient number [2, 0], 2 if one_num
           Other edge [d, a, ...], orient number [3, 0], 3 if one_num
           Other edge [a, d, ...], orient number [1, 0], 4 if one_num
           Other edge [b, a, ...], orient number [1, 1], 5 if one_num
           Other edge [c, b, ...], orient number [1, 2], 6 if one_num
           Other edge [d, c, ...], orient number [1, 3], 7 if one_num
        For in-depth discussion, see
        https://github.mit.edu/mirabito/MSEAS-3DHDG/issues/30
        """
        # label the v1 vertices [a, b, c, d] and find the positions of the v2
        # vertices that correspond to [a] and [b]
        relative_positions = [(v1[i] == v2).nonzero()[0][0] for i in range(2)]
        enumeration = self.orient_enumeration[tuple(relative_positions)]
        return enumeration


class Mesh2D(Mesh):
    elm_enumeration = {"TRIANGLE": 0, "QUAD": 1}

    # face / edge orientation enumeration as discussed in-depth in
    # https://github.mit.edu/mirabito/MSEAS-3DHDG/issues/30
    orient_enumeration = {(0, 1): 0, (1, 0): 1}

    def __init__(self, elm, vert):
        r"""
        We are assuming only two types of elements here. 2D triangles and 2D
        quadrilaterals.
        @param elm (\c int) Numpy array that defines the elements. Each row
                    is an element, and each column indicates a vertex of the
                    element. That is elm[0, 0] gives the global vertex number
                    of vertex 0 of element 0.
        @param vert (\c float) Numpy array that gives the spatial coordinates
                    of the global vertices.
        @note The elm and vert inputs may be inputted as a tuple, for
              convenience.
        @code
        >>> import msh
        >>> t,p = msh.mk.struct2D()
        >>> mesh = msh.Mesh2D(t,p)
        >>> #OR equivalently
        >>> mesh = msh.Mesh2D(msh.mk.struct2D())
        @endcode
        @author Matt Ueckermann
        """
        dim = 2
        elm, vert = fix(elm, vert)  # CCW ordering of element nodes
        elm = mshu.sort_by_element_type(elm)  # Tri first, then quads

        n_elm = len(elm)
        n_tri = (elm[:, 3] < 0).sum()
        # convenience attributes
        self.n_tri = n_tri
        self.n_quad = n_elm - n_tri

        n_elm_type = [n_tri, n_elm - n_tri]
        if n_tri == 0:     n_elm_type = [n_elm]  # all quads
        if n_elm == n_tri: n_elm_type = [n_elm]  # all tris

        # Now sort by vertex number
        if n_tri > 0: elm[0:n_tri, :] = mshu.sort_by_vertex_number(elm[0:n_tri, :])
        if n_tri < n_elm: elm[n_tri:n_elm, :] = mshu.sort_by_vertex_number(
            elm[n_tri:n_elm, :])

        elm_type = np.zeros(n_elm, dtype=int)
        if n_tri == n_elm:
            u_elm_type = [self.elm_enumeration["TRIANGLE"]]
        elif n_tri == 0:
            u_elm_type = [self.elm_enumeration["QUAD"]]
        else:
            u_elm_type = [self.elm_enumeration["TRIANGLE"],
                          self.elm_enumeration["QUAD"]]
            elm_type[n_tri:n_elm] = self.elm_enumeration["QUAD"]

        # create the connectivity matrixes (Which needs the element enumerated type)
        elm2elm, ed2ed = connect_elm(elm, np.array(u_elm_type)[elm_type], dim, u_elm_type)

        ## The element connectivity matrix. elm2elm[i, j] gives the element
        # number which is connected to element i, through edge j of element i.
        self.elm2elm = elm2elm

        ## The edge connectivity matrix.
        # ed2ed[i, 0:2] gives the [element #, local edge #] of the plus-side element.
        # ed2ed[i, 2:4] gives the [element #, local edge #] of the minus-side element.
        # ed2ed[i, 4:] gives the vertices that make up the edge.
        # numbered CCW with outward-point normal (according to Right-hand rule)
        # CF: This seems to indicate that left is plus, right is minus
        self.ed2ed = ed2ed

        ##A boolean array used to select the interior edges only
        self.ids_interior_ed = (self.ed2ed[:, 2] >= 0).ravel()

        ##A boolean array used to select the exterior edges only
        self.ids_exterior_ed = (ed2ed[:, 2] < 0).ravel()

        ## The triangulation matrix that defined each element.
        # elm[i, :] gives the global vertex numbers that make up the element.
        # This matrix is ordered such that the first num2Dtri elements are
        # triangular elements, while the remaining ones are quadrilaterals.
        self.elm = elm

        ## The different unique types of elements in the triangulation.
        self.u_elm_type = np.array(u_elm_type, dtype=int)

        ## The element type. u_elm_type[elm_type[i]] gives the type of element
        # for global element number i.
        self.elm_type = elm_type

        ## The edge element type. u_ed_type[elm_type[i]] gives the type of
        # edge element for global edge element number i. For 2D, there is only
        # one edge type -- lines
        self.ed_type = np.zeros(len(self.ed2ed), dtype=int)

        ## The different unique types of edges in the triangulation. for 2D
        # there is only the one type -- lines
        self.u_ed_type = np.array([0], dtype=int)

        ## Gives the total number of elements in the triangulation.
        # The number of triangles is given by n_tri, and the number of
        # quads can be calculated using n_elm-n_tri
        self.n_elm = n_elm

        ## Gives the number of elements of a particular type in the triangulation
        self.n_elm_type = n_elm_type

        ## Gives the total number of edges in the triangulation.
        self.n_ed = len(self.ed2ed)

        ## Gives the number of edge elements of a particular type
        self.n_ed_type = [len(self.ed2ed)]

        # Array giving the x-y coordinates of the global vertices in the
        # triangulation.
        self.vert = vert

        # The dimension of the mesh, dim=2, since this Mesh2D is exclusively for
        # 2D meshes.
        self.dim = dim

        ##Vertex map, maps the vertex number from one periodic edge to the
        # other. This map is needed when comparing the orientation of the edge
        # on the element to the orientation of the periodic edge. The element on
        # the right will not have matching vertex numbers, because it's edge
        # used to be a boundary edge, but has disappeared because of the
        # periodicity.
        # EG. in 1D:
        # [0] a1--(A)--a0 1 b0--(B)--b1 [2] ==> 0 --(A)-- 1 --(B)-- 0
        # ed2ed = [A a0  B  b0 1                   ed2ed = [A a0 B b0 1
        #          A a1 -1 -1  0            ==>             A a1 B b1 0]
        #          B b1 -1 -1  2]
        # elm = [0 1                        ==>      elm = [0 1
        #        1 2]                                       1 2]
        #
        # This array is populated in the msh.mk.periodic function
        self.vertmap = None

        # Next we have to build a rather annoying structure -- the elements have
        # global numbers -- however the data is stored/organized according to the
        # element type. So within the element type, the element will have a
        # different number/location. The next structure figures out what that
        # element number is. The same goes for the edge types
        ## The "global element number" to "element number within a type"
        # conversion array. For example, the data for global element number i
        # is stored in field[elm_type[i]][:, :, glob2type_elm[i]].
        self.glob2type_elm = np.zeros(self.n_elm, dtype=int)
        sumtype = [0] * len(self.u_elm_type)
        for i in range(self.n_elm):
            elm_type = self.elm_type[i]
            self.glob2type_elm[i] = sumtype[elm_type]
            sumtype[elm_type] += 1

        ## The "element number within a type" to "global element number" list

        ## The "global edge number" to "edge number within a type"
        # conversion array. For example, the data for global edge number i
        # is stored in field_ed[ed_type[i]][:, :, glob2type_ed[i]].
        self.glob2type_ed = np.zeros(self.n_ed, dtype=int)
        sumtype = [0] * len(self.u_ed_type)
        for i in range(self.n_ed):
            ed_type = self.ed_type[i]
            self.glob2type_ed[i] = sumtype[ed_type]
            sumtype[ed_type] += 1

        ##A list of boolean arrays used to select the interior edges only
        self.ids_interior_ed_by_type = [self.ids_interior_ed[self.ed_type == i] \
                                        for i in range(len(self.n_ed_type))]

        ##A list of boolean arrays used to select the exterior edges only
        self.ids_exterior_ed_by_type = [self.ids_exterior_ed[self.ed_type == i] \
                                        for i in range(len(self.n_ed_type))]

        ##Index mapping array from ed_type edge id number to ed_bc_type id
        # number. Basically, in the solver we will refer to, for e.g. the
        # data field_ed[i][:, :, j], where j refers to a boundary edge, numbered
        # according to the element-type local id number. The boundary condition
        # data is stored in an array smaller that field_ed, that is, field_ed_bc
        # contains ONLY the boundary condition information, so calling
        # field_ed_bc[i][:, :, j] will exceed the array bounds. Instead we call
        # field_ed_bc[i][:, :, in2ex_bcid[j]].
        # TODO: Determine if this array is actually still needed
        #   (Indexing has been improved since the below was implemented)
        self.in2ex_bd_id = [ex.cumsum() - 1 for ex in self.ids_exterior_ed_by_type]

        ## CF: experimental -- untested
        self.elm2ed = connect_elm2ed(self.elm2elm, self.ed2ed)
        self.type2glob_elm = self.build_type2glob_elm_list()

        # build a set of elements on the boundary for fast lookup
        self.boundary_elms = self.build_boundary_elm_sets()

    def connectivity_by_elm_type(self, elm_type):
        """ returns the connectivity list for a single element type """
        conn_mask = self.elm_type == self.elm_enumeration[elm_type]
        if elm_type == "TRIANGLE":
            return self.elm[conn_mask, :3]  # ignore -1
        elif elm_type == "QUAD":
            return self.elm[conn_mask, :]
        else:
            raise ValueError("elm_type not understood")

    def edge_vertex_numbers(self):
        """ returns the edge vertex numbers in the mesh """
        return self.ed2ed[:, 4:]

    def edge_vertices(self):
        """ returns the points of all the edge vertices in the mesh
        @retval ed_verts  (n_edges, dim, local vertex number (0, 1))
        """
        ed = self.edge_vertex_numbers()
        return self.vert[:, :2][ed]

    def fix(self):
        ''' Function that ensures the the elements are properly numbered in
        a counter-clockwise fashion, with no crosses. This function updates the
        elm and vert data members.
        @see msh.util.fix
        '''
        self.elm, self.vert = fix(self.elm, self.vert)

    def vol(self, ids=None):
        return simpvol2D(self.elm, self.vert) if ids is None else simpvol2D(self.elm[ids, :], self.vert)

    def set_bc_ids(self, bc_id_lambda, labels=None):
        r"""To change the default id number for boundary conditions, you can
        use this function
        @param bc_id_lambda (\c lambda function) List of lambda functions. The
                            id of the list determines the id of the boundary.
               bc_id_lambda = lambda (ctrd): f(ctrd)
               where ctrd is a numpy array with ctrd.shape = (n_ext_ed, dim) with the
               centroids of the edges. bc_id_lambda[i](ctrd) should evaluate to
               True if that edge should have the id '-i-1'.
        :param labels: optional list of string labels ordered like bc_id_lambda
        CF: Note that the list of boundaries are traversed in the order they
        occur in the list bc_id_lambda, so the final ID is the LAST index of
        the containing the lambda function which returns true when called on
        the edge centroid.
        """
        # Find edge centroids
        ids = (self.ids_interior_ed == False).nonzero()[0]
        vts = self.ed2ed[ids, 4:]
        ctrd = np.array([coord[vts].mean(1) for coord in self.vert[:].T]).T

        for i in range(len(bc_id_lambda)):
            self.ed2ed[ids[bc_id_lambda[i](ctrd)], 2:3] = -i - 1

        # Boundary condition information
        tot_bc_ids = -min(self.ed2ed[self.ids_exterior_ed, 2])

        ##Total number of different bc_ids
        self.n_bc_id = tot_bc_ids

        if labels:
            bc_ids = np.arange(len(bc_id_lambda))
            self.bcid_labels = {_str: idx for _str, idx in zip(labels, bc_ids)}

    def write_mesh_to_vtk(self, filename):
        """
        author: CF
        write the 2D mesh out to VTK file so that it can be viewed in Paraview
        or some similar software
        """

        pts, conn = self.vert, self.elm
        Points, Cells = vtk.vtkPoints(), vtk.vtkCellArray()

        # add node / connectivity information to VTK object
        for pt in pts:
            Points.InsertNextPoint(pt)

        for cn in conn:
            cell = vtk.vtkTriangle() if cn[-1] == -1 else vtk.vtkQuad()
            for idx, pt in enumerate(cn):
                if pt != -1:
                    cell.GetPointIds().SetId(idx, pt)
            Cells.InsertNextCell(cell)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(Points)
        polydata.SetPolys(Cells)

        # write VTK object to file
        polydata.Modified()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polydata.Update()

        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName(filename);
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()


def refine_mesh(mesh2D):
    """ refines a mesh2D instance, returning another
    """
    T, P = np.copy(mesh2D.elm), np.copy(mesh2D.vert)
    ntri, nquad = mesh2D.n_tri, mesh2D.n_quad

    ref_T = []
    # triangular mesh refinement
    if ntri > 0:
        tri_T = T[0:ntri, :]
        refTriT, P = refine_triangular_mesh(tri_T, P)
        ones_pad = -1 * np.ones((len(refTriT), 1), dtype=np.int)
        ref_T.append(np.hstack((refTriT, ones_pad)))

    # quad mesh refinement
    if nquad > 0:
        quad_T = T[-nquad:, :]
        refQuadT, refP = refine_quad_mesh(quad_T, P)
        ref_T.append(refQuadT)
        P = refP
    T = np.vstack(ref_T)
    refMesh = Mesh2D(T, P)
    return refMesh


def refined_tri_connectivity(n, T):
    """computes refined connectivity array
    @param n the current number of vertices in mesh
    @param T the connectivity of the current triangle
    """
    v0, v1, v2 = T
    v3, v4, v5 = n, n + 1, n + 2
    ref_T = np.array([[v0, v3, v5],
                      [v3, v1, v4],
                      [v5, v4, v2],
                      [v3, v4, v5]])
    return ref_T


def refine_triangle_p(p):
    """refines a triangle with vertices specified by p
    @param p  ndarray of triangle vertices shape (3, 3), (vert, xyz)
    @retval p_ref  ndarray of pts on the refined triangle (6, 3)
    """
    refine_pts = list(p)
    for vert1, vert2 in [(0, 1), (1, 2), (2, 0)]:
        midpoint = 0.5 * refine_pts[vert1] + 0.5 * refine_pts[vert2]
        refine_pts.append(midpoint)
    p_ref = np.vstack(refine_pts)
    return p_ref


def split_triangle(tri_P, split_vertex):
    '''takes in triangle's T and P arrays, as well as the vertex to split,
    returns new P array of new triangles
    @param tri_p  ndarray of triangle vertices shape (3,3), (vert, xyz)
    @param split_vertex  index of vertice to split the triangle on
    @retval ndarray of pts on the refined triangle (4, 3)
    '''
    refine_pts = list(tri_P)
    temp = list()
    for vert in range(len(tri_P)):
        if vert != split_vertex:
            temp.append(vert)
    midx = 0.5 * refine_pts[temp[0]][0] + 0.5 * refine_pts[temp[1]][0]
    midy = 0.5 * refine_pts[temp[0]][1] + 0.5 * refine_pts[temp[1]][1]
    midpoint = [midx, midy, 0]
    midpoint = np.array(midpoint)
    return midpoint


def split_tri_connectivity(n, tri_T, split_vertex):
    """computes the split triangle connectivity array
    @param n the current_number of vertices in the mesh
    @param tri_T the connectivity of the current triangle
    @param split_vertex index of vertex to split
    @retval ref_T
    """
    v0, v1, v2, null = tri_T
    if split_vertex == 0:
        ref_T = np.array([[v0, n, v2, -1],
                          [v0, v1, n, -1]])
    if split_vertex == 1:
        ref_T = np.array([[v0, v1, n, -1],
                          [n, v1, v2, -1]])
    if split_vertex == 2:
        ref_T = np.array([[v0, n, v2, -1],
                          [n, v1, v2, -1]])
    return ref_T


def find_longest_edge_verts_tri(tri_P):
    """takes in P array and finds the longest edge of the triangle; returns verts of longest edge
    @param tri_P P array of the triangle
    @retval returns the vertices of the longest edge
    """
    dist = []
    verts = [(0, 1), (1, 2), (2, 0)]
    for vert1, vert2 in verts:
        distance = ((abs(tri_P[vert1][0] - tri_P[vert2][0]) ** 2) + (
                    abs(tri_P[vert1][1] - tri_P[vert2][1]) ** 2)) ** 0.5
        dist.append(distance)
    longest = max(dist)
    for i in range(len(dist)):
        if dist[i] == longest:
            return verts[i]


def conforming_refinement(mesh, elements_to_refine):
    """
    :param mesh: Mesh2D instance containing only triangular elements
    :elements_to_refine: list containing elements that should be refined
    """

    # turn mesh vertices into a list of np.vectors for each point
    # start an empty list for the new connectivities

    T_ori, P_ori = np.copy(mesh.elm), np.copy(mesh.vert)
    p_list = list(P_ori)

    T_refined, P_refined = refine_mesh_with_split(T_ori, P_ori, elements_to_refine)

    # for each element in mesh
    # if element is to be refined,
    # split element, check neighbors, refine neighbors if necessary
    # building a new T_refined / P_refined array as you go
    #
    # if tri is not to be refined or touched, just add its connectivity
    # to the T_refined list as it is

    # stack the list into two single arrays to finish things up

    return T_refined, P_refined


def refine_mesh_with_split(tri_T, P, refine_indices):
    """takes in T and P arrays, and boolean array of tri to refine; returns refined T and P
    @param P ndarray, mesh.verts, constains all mesh points (tri and quad)
    @param tri_T  the connectivity array of all mesh tris (ntris, 4)
    @param bool_array a list of indices for refinement
    """
    ref_T, ref_P, tri_T = list(tri_T), list(P), list(tri_T)
    current_mesh_verts = len(ref_P)
    total_refined = 0
    collateral = []

    # iterate through elements to be refined
    for index in sorted(refine_indices):
        new_index = index - total_refined

        # set up triangle verts and find the verts of the longest edge of the triangle, to find
        # the split vertex
        conn = ref_T[new_index]
        tri_verts = list()
        for i in conn[:3]:
            tri_verts.append(ref_P[i])

        longest_edge_indices = find_longest_edge_verts_tri(tri_verts)
        split_index = 0
        similar_verts = []
        for vert in range(len(conn[:3])):
            if vert not in longest_edge_indices:
                split_index = vert
            else:
                similar_verts.append(conn[vert])
        split_vertex = conn[split_index]

        # create new vertex
        new_vert = split_triangle(tri_verts, split_index)
        ref_P.append(new_vert)

        # create new connectivity for split triangle
        new_conn = list(split_tri_connectivity(current_mesh_verts, conn, split_index))
        ref_T.append(new_conn[0])
        ref_T.append(new_conn[1])

        # split connected triangle along same vertice, if it exists

        # first find opposite triangle (if it exists), then opposite vertex
        opp_tri = list()
        opp_tri_bool = False
        opp_tri_index = 0
        for i in range(len(ref_T)):
            tri = list(ref_T[i])[:3]
            if similar_verts[0] in tri and similar_verts[1] in tri and split_vertex not in tri:
                opp_tri = list(ref_T[i])
                opp_tri_bool = True
                opp_tri_index = i
                break
        if opp_tri_bool:
            opp_split_index = 0
            for i, vert in enumerate(opp_tri[:3]):
                if vert not in similar_verts:
                    opp_split_index = i

                    # create new connectivity for opposite triangle (vertex already added to P)
            opp_new_conn = list(split_tri_connectivity(current_mesh_verts, opp_tri, opp_split_index))
            ref_T.append(opp_new_conn[0])
            ref_T.append(opp_new_conn[1])

            #             import pdb; pdb.set_trace()
            # delete old connectivities
            if opp_tri_index > new_index:
                del ref_T[opp_tri_index]
                del ref_T[new_index]
            else:
                del ref_T[new_index]
                del ref_T[opp_tri_index]

            # determine if oposing triangle was prerefined or not
            prerefined_tri = False
            for ori_tri in tri_T:
                if opp_tri[0] == ori_tri[0] and opp_tri[1] == ori_tri[1] and opp_tri[2] == ori_tri[2]:
                    prerefined_tri == True
                    break
            if prerefined_tri:
                total_refined += 2
            else:
                total_refined += 1

        else:
            #             import pdb; pdb.set_trace()
            del ref_T[new_index]
            total_refined += 1

        # update mesh vertices
        current_mesh_verts += 1

    # create the new T,P arrays
    ref_P, ref_T = np.vstack(ref_P), np.vstack(ref_T)
    return ref_T, ref_P


def refine_triangular_mesh(tri_T, P):
    """takes P array of the mesh+tri connectivity; returns refined T, P
    @param P ndarray, mesh.verts, contains all mesh points (tri and quad)
    @param tri_T  the connectivity array of all mesh tris (ntris, 4)
    """
    P, tri_T = P[:, :2], tri_T[:, :3]
    ref_T, new_P = list(), [P, ]
    current_mesh_verts = len(P)
    for i, conn in enumerate(tri_T):
        tri_verts = P[conn, :]

        # create new vertices
        new_verts = refine_triangle_p(tri_verts)[3:]
        new_P.append(new_verts)

        # create new connectivity for this triangle
        new_conn = refined_tri_connectivity(current_mesh_verts, conn)
        ref_T.append(new_conn)

        # update mesh vertices
        current_mesh_verts += 3

    # create the new T,P arrays
    ref_P, ref_T = np.vstack(new_P), np.vstack(ref_T)
    return ref_T, ref_P


def refined_quad_connectivity(n, T):
    """computes refined connectivity array
    @param n the current number of vertices in mesh
    @param T the connectivity of the current quad
    @note, v8  refers to the new center point of the quad
    """
    v0, v1, v2, v3 = T
    v4, v5, v6, v7, v8 = n, n + 1, n + 2, n + 3, n + 4
    ref_T = np.array([[v0, v4, v8, v7],
                      [v4, v1, v5, v8],
                      [v8, v5, v2, v6],
                      [v7, v8, v6, v3]])
    return ref_T


def refine_quad_p(p):
    """refines a quad with vertices specified by p, CCW
    @param p  ndarray of quad vertices shape (4, 3), (vert, xyz)
    @retval p_ref  ndarray of pts on the refined triangle (5, 3)
    """
    refine_pts = list(p)
    for vert1, vert2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        midpoint = 0.5 * refine_pts[vert1] + 0.5 * refine_pts[vert2]
        refine_pts.append(midpoint)
    refine_pts.append(quad_center_of_mass(p))
    p_ref = np.vstack(refine_pts)
    return p_ref


def quad_center_of_mass(quadpts):
    """ computes center of mass of the quad
    @param quadpts  (4,2) array of quad verts, CCW
    """
    return np.mean(quadpts, axis=0)


def refine_quad_mesh(quad_T, P):
    """takes P array of the mesh+quad connectivity; returns refined T, P
    @param P ndarray, mesh.verts[:,:2], contains all mesh pts
    @param quad_T  the connectivity array of all mesh quads (nquads, 4)
    """
    ref_T, new_P = list(), [P, ]
    current_mesh_verts = len(P)
    for i, conn in enumerate(quad_T):
        quad_verts = P[conn, :]

        # create new vertices
        new_verts = refine_quad_p(quad_verts)[4:]
        new_P.append(new_verts)

        # create new connectivity for this triangle
        new_conn = refined_quad_connectivity(current_mesh_verts, conn)
        ref_T.append(new_conn)

        # update mesh vertices
        current_mesh_verts += 5

    # create the new T,P arrays
    ref_P, ref_T = np.vstack(new_P), np.vstack(ref_T)
    return ref_T, ref_P


import math
import sys

import numpy as np

import pyclipper

import mandoline.mesh as msh


SCALING_FACTOR = 1000


def offset(paths, amount):
    pco = pyclipper.PyclipperOffset()
    pco.ArcTolerance = SCALING_FACTOR / 40
    paths = pyclipper.scale_to_clipper(paths, SCALING_FACTOR)
    pco.AddPaths(paths, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
    outpaths = pco.Execute(amount * SCALING_FACTOR)
    outpaths = pyclipper.scale_from_clipper(outpaths, SCALING_FACTOR)
    return outpaths


def union(paths1, paths2):
    if not paths1:
        return paths2
    if not paths2:
        return paths1
    pc = pyclipper.Pyclipper()
    if paths1:
        if paths1[0][0] in (int, float):
            raise pyclipper.ClipperException()
        paths1 = pyclipper.scale_to_clipper(paths1, SCALING_FACTOR)
        pc.AddPaths(paths1, pyclipper.PT_SUBJECT, True)
    if paths2:
        if paths2[0][0] in (int, float):
            raise pyclipper.ClipperException()
        paths2 = pyclipper.scale_to_clipper(paths2, SCALING_FACTOR)
        pc.AddPaths(paths2, pyclipper.PT_CLIP, True)
    try:
        outpaths = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    except:
        print("paths1={}".format(paths1))
        print("paths2={}".format(paths2))
    outpaths = pyclipper.scale_from_clipper(outpaths, SCALING_FACTOR)
    return outpaths


def diff(subj, clip_paths, subj_closed=True):
    if not subj:
        return []
    if not clip_paths:
        return subj
    pc = pyclipper.Pyclipper()
    if subj:
        subj = pyclipper.scale_to_clipper(subj, SCALING_FACTOR)
        pc.AddPaths(subj, pyclipper.PT_SUBJECT, subj_closed)
    if clip_paths:
        clip_paths = pyclipper.scale_to_clipper(clip_paths, SCALING_FACTOR)
        pc.AddPaths(clip_paths, pyclipper.PT_CLIP, True)
    outpaths = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    outpaths = pyclipper.scale_from_clipper(outpaths, SCALING_FACTOR)
    return outpaths


def clip(subj, clip_paths, subj_closed=True):
    if not subj:
        return []
    if not clip_paths:
        return []
    pc = pyclipper.Pyclipper()
    if subj:
        subj = pyclipper.scale_to_clipper(subj, SCALING_FACTOR)
        pc.AddPaths(subj, pyclipper.PT_SUBJECT, subj_closed)
    if clip_paths:
        clip_paths = pyclipper.scale_to_clipper(clip_paths, SCALING_FACTOR)
        pc.AddPaths(clip_paths, pyclipper.PT_CLIP, True)
    out_tree = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    outpaths = pyclipper.PolyTreeToPaths(out_tree)
    outpaths = pyclipper.scale_from_clipper(outpaths, SCALING_FACTOR)
    return outpaths


def paths_contain(pt, paths):
    cnt = 0
    pt = pyclipper.scale_to_clipper([pt], SCALING_FACTOR)[0]
    for path in paths:
        path = pyclipper.scale_to_clipper(path, SCALING_FACTOR)
        if pyclipper.PointInPolygon(pt, path):
            cnt = 1 - cnt
    return cnt % 2 != 0


def orient_path(path, dir):
    orient = pyclipper.Orientation(path)
    path = pyclipper.scale_to_clipper(path, SCALING_FACTOR)
    if orient != dir:
        path = pyclipper.ReversePath(path)
    path = pyclipper.scale_from_clipper(path, SCALING_FACTOR)
    return path


def orient_paths(paths):
    out = []
    while paths:
        path = paths.pop(0)
        path = orient_path(path, not paths_contain(path[0], paths))
        out.append(path)
    return out


def paths_bounds(paths):
    if not paths:
        return (0, 0, 0, 0)
    minx, miny = (None, None)
    maxx, maxy = (None, None)
    for path in paths:
        for x, y in path:
            if minx is None or x < minx:
                minx = x
            if maxx is None or x > maxx:
                maxx = x
            if miny is None or y < miny:
                miny = y
            if maxy is None or y > maxy:
                maxy = y
    bounds = (minx, miny, maxx, maxy)
    return bounds


def close_path(path):
    if not path:
        return path
    if path[0] == path[-1]:
        return path
    return path + path[0:1]


def close_paths(paths):
    return [close_path(path) for path in paths]


############################################################


def make_infill_pat(rect, baseang, spacing, rots):
    minx, miny, maxx, maxy = rect
    w = maxx - minx
    h = maxy - miny
    cx = math.floor((maxx + minx)/2.0/spacing)*spacing
    cy = math.floor((maxy + miny)/2.0/spacing)*spacing
    r = math.hypot(w, h) / math.sqrt(2)
    n = int(math.ceil(r / spacing))
    out = []
    for rot in rots:
        c1 = math.cos((baseang+rot)*math.pi/180.0)
        s1 = math.sin((baseang+rot)*math.pi/180.0)
        c2 = math.cos((baseang+rot+90)*math.pi/180.0) * spacing
        s2 = math.sin((baseang+rot+90)*math.pi/180.0) * spacing
        for i in range(1-n, n):
            cp = (cx + c2 * i, cy + s2 * i)
            line = [
                (cp[0] + r  * c1, cp[1] + r * s1),
                (cp[0] - r  * c1, cp[1] - r * s1)
            ]
            out.append( line )
    return out


def make_infill_lines(rect, base_ang, density, ewidth):
    if density <= 0.0:
        return []
    if density > 1.0:
        density = 1.0
    spacing = ewidth / density
    return make_infill_pat(rect, base_ang, spacing, [0])


def make_infill_triangles(rect, base_ang, density, ewidth):
    if density <= 0.0:
        return []
    if density > 1.0:
        density = 1.0
    spacing = 3.0 * ewidth / density
    return make_infill_pat(rect, base_ang, spacing, [0, 60, 120])


def make_infill_grid(rect, base_ang, density, ewidth):
    if density <= 0.0:
        return []
    if density > 1.0:
        density = 1.0
    spacing = 2.0 * ewidth / density
    return make_infill_pat(rect, base_ang, spacing, [0, 90])


def make_infill_hexagons(rect, base_ang, density, ewidth):
    if density <= 0.0:
        return []
    if density > 1.0:
        density = 1.0
    ext = 0.5 * ewidth / math.tan(60.0*math.pi/180.0)
    aspect = 3.0 / math.sin(60.0*math.pi/180.0)
    col_spacing = ewidth * 4./3. / density
    row_spacing = col_spacing * aspect
    minx, maxx, miny, maxy = rect
    w = maxx - minx
    h = maxy - miny
    cx = (maxx + minx)/2.0
    cy = (maxy + miny)/2.0
    r = max(w, h) * math.sqrt(2.0)
    n_col = math.ceil(r / col_spacing)
    n_row = math.ceil(r / row_spacing)
    out = []
    s = math.sin(base_ang*math.pi/180.0)
    c = math.cos(base_ang*math.pi/180.0)
    for col in range(-n_col, n_col):
        path = []
        base_x = col * col_spacing
        for row in range(-n_row, n_row):
            base_y = row * row_spacing
            x1 = base_x + ewidth/2.0
            x2 = base_x + col_spacing - ewidth/2.0
            if col % 2 != 0:
                x1, x2 = x2, x1
            path.append((x1, base_y+ext))
            path.append((x2, base_y+row_spacing/6-ext))
            path.append((x2, base_y+row_spacing/2+ext))
            path.append((x1, base_y+row_spacing*2/3-ext))
        path = [(x*c - y*s, x*s + y*c) for x, y in path]
        out.append(path)
    return out


def make_infill_variable(rect, layer_stress, ewidth):
    minx, miny, maxx, maxy = rect
    ori_T = np.array([[0, 1, 2, 3]])
    ori_P = np.array([[minx, miny, 0],
             [maxx, miny, 0],
             [maxx, maxy, 0],
             [minx, maxy, 0]])

    # ori_P = np.array([[0, 0, 0],
    #                   [5, 0, 0],
    #                   [5, 5, 0],
    #                   [0, 5, 0]])

    ori_mesh = msh.Mesh2D(elm=ori_T, vert=ori_P)
    refine_mesh = refine_layer(0.1, 0.9, ori_mesh, layer_stress, ewidth)
    sys.exit(-1)
    # raise(ValueError, "No Infill Pattern Implemented")


def refine_layer(min_dense, max_dense, mesh, layer_stress, ewidth):
    T, V = np.copy(mesh.elm), np.copy(mesh.vert)
    to_refine = []

    min_sp = density2space(max_dense, ewidth)
    max_sp = density2space(min_dense, ewidth)

    # Iterate through each quad element in the mesh
    for i in range(T.shape[0]):
        quad = T[i, :]

        # Get the x and y points for this quad
        x_points = [v[0] for v in [V[i] for i in quad]]
        y_points = [v[1] for v in [V[i] for i in quad]]
        maxx, minx = max(x_points), min(x_points)
        maxy, miny = max(y_points), min(y_points)

        # Refine quad too large of a distance between points
        if any([maxx-minx > max_sp, maxy-miny > max_sp]):
            to_refine.append(quad)
        else:
            # Get the acceptable normalized stress threshold
            density = space2density(maxx - minx, ewidth)
            stress_accept = (density-min_dense)/(max_dense-min_dense)

            # Sample the stress, and refine if it is too big
            max_stress = sample_stress(layer_stress, (minx, miny, maxx, maxy))
            if max_stress > stress_accept:
                to_refine.append(quad)

        if to_refine:
            new_mesh = msh.non_conforming_refinement(mesh, to_refine)
            return refine_layer(min_dense, max_dense, new_mesh, layer_stress, ewidth)

        return mesh


def sample_stress(layer_stress, rect):
    "Return max stress in a given rectangle of a stress layer"
    minx, miny, maxx, maxy = rect
    max_stress = 0

    for i, coords in enumerate(layer_stress['coords']):
        x, y = coords[0], coords[1]
        n_stress = layer_stress['stress'][i]
        if minx <= x <= maxx and miny <= y <= maxy and n_stress > max_stress:
            max_stress = n_stress

    return max_stress


def space2density(spacing, ewidth):
    return 2.0 * ewidth / spacing


def density2space(density, ewidth):
    return 2.0 * ewidth / density

# vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap

import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

data_limits_msg = \
"""Not all data limits are specified. Each patch is being plotted from colormap
min to max rather than all patches plotted according to the color limits of the data.
"""

################################################################################
# Patch plotting
################################################################################
def scalar_plot(ax, nodes, field, ensemble=0, *args, cmap=mpl.cm.viridis,
                            colorbar=True, **kwargs):
    """ high level plotting API for plotting a 2D DG field
    patch plot for ScalarField object
    :param ax: the matplotlib axis object
    :param nodes: PositionField instance for an interior field
    :param data: sol.trace-like data indexed by field, i.e., sol.trace[:,field,:]
    :param height_plot: bool to denote whether a 3d height plot is to be
        produced instead of a flat patch plot; in this case, ax must be set with 'projection=3d'
    example:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        scalar_field_patch_plot(ax, position_field, scalar_field,
                                cmap=cmocean.cm.balance, shading='gouraud')
    """
    # compute or manually set data limits with respect to colormap
    dataMax, dataMin = max(np.max(arr) for arr in field), min(np.min(arr) for arr in field)
    if 'vmax' in kwargs: dataMax = kwargs.pop('vmax')
    if 'vmin' in kwargs: dataMin = kwargs.pop('vmin')

    # plot the scalar field for each element type
    for idx, arr in enumerate(field):
        ax = etype_patch_plot(ax, nodes[idx], arr[ensemble,...],
                vmax=dataMax, vmin=dataMin, cmap=cmap, *args, **kwargs)
    # colorbar
    if colorbar:
        create_standard_colorbar(ax, dataMin, dataMax, cmap)
    return ax

def etype_patch_plot(ax, dgnodes, data, *args, **kwargs):
    """ patch plots for a given element type, leveraging draw_element
    :param dgnodes: np array with nodal positions of a single element type (n_elms, 2, nodes)
    :param data:  np array with data corresponding to dgnodes, (n_elm, nodes)
    :note: draws on the axis object and returns, args, kwargs passed through to draw_element
    :note: will draw each element without respect to max/min data. This is
        usually passed through a higher-level call but can be manually specified
        with vmax and vmin args.
    """
    if 'vmax' not in kwargs or 'vmin' not in kwargs: warnings.warn(data_limits_msg)
    n_elm, dim, n_nodes = dgnodes.shape
    for elm in range(n_elm):
        nodes = dgnodes[elm, ...]
        pltData  = data[elm, ...]
        draw_element(ax, nodes, pltData, *args, **kwargs)
    return ax

def draw_element(ax, nodes, data, height_plot=False, *args, **kwargs):
    """" triangulates data over element and plots result on ax
    :param ax:  matplotlib Axis object
    :param nodes:  nodes at which to plot the data shape (2, nodes)
    :param data:  np vector of values of the scalar field to plot shape (nodes,)
    :note: function receives optional arguments to tripcolor such as shading, data limits, colormap,
    etc
    """
    # create a triangulation in matplotlib and remove flat elements, see
    #https://github.mit.edu/mirabito/MSEAS-3DHDG/issues/43
    elmTri = tri.Triangulation(nodes[0, :], nodes[1, :])
    mask = tri.TriAnalyzer(elmTri).get_flat_tri_mask(min_circle_ratio=0.01)
    elmTri.set_mask(mask)

    if height_plot:
        plot = ax.plot_trisurf(elmTri, data, *args, **kwargs)
    else:
        plot = ax.tripcolor(elmTri, data, *args, **kwargs)
    return ax

def create_colorbar(data, ax, cmap, dataMin, dataMax,
        size="4%", pad=0.2, *args,**kwargs):
    """ adds a colorbar to an axis ax
    @param data numpy array containing the data represented by the colorbar
    @param ax  the axis on which to draw the colorbar
    @param cmap  the colormap for the colorbar
    @param dataMin, dataMax  floats specifying the limits of the colorbar
    """
    # create a new colorbar and apply it to our shrunken axis due to the non square domain, this is
    # necessary to scale the colorbar correctly.
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size=size, pad=pad)

    # set the current axis back to the main axis, otherwise current axis is directed to the other
    # subplot colorbar, and the new colorbar is super super small
    plt.sca(ax)
    midColorBarTick = np.mean([dataMin, dataMax])
    norm = mpl.colors.Normalize(vmin=dataMin, vmax=dataMax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(data)
    plt.colorbar(m, cax=color_axis, ticks=[dataMin, midColorBarTick, dataMax], **kwargs)

def create_standard_colorbar(ax, dataMin, dataMax, cmap=mpl.cm.viridis, size="4%", pad=0.2):
    """
    Creates a simple colorbar using dataMin, dataMax limits
    """
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size=size, pad=pad)
    norm = mpl.colors.Normalize(vmin=dataMin, vmax=dataMax)
    colorbar = mpl.colorbar.ColorbarBase(ax=color_axis, norm=norm, cmap=cmap)#format='%.1e')

def draw_polygon(pts, ax, *args, **kwargs):
    """ draws a polygonal element from a mesh
    @param pts (poly_verts, 2) array of vertices, CCW
    @param ax  matplotlib axis object to which to add the polygon
    @param *args, **kwargs, arguments to the mpl.patches.Polygon class
    """
    p = mpl.patches.Polygon(pts[:,:2], fill=False, *args, **kwargs)
    ax.add_patch(p)
    return ax

def draw_mesh(mesh, ax, color='k', scatter=True, *args, **kwargs):
    """draws a 2D mixed mesh
    @param mesh  mesh2D instance
    @param ax  matplotlib axis object on which to plot
    """
    draw_numbers = False
    if 'annotate_elms' in kwargs:
        kwargs.pop('annotate_elms')
        draw_numbers = True

    for i, conn in enumerate(mesh.elm):
        # if tri, remove bogus -1 connectivity pt
        if conn[-1] == -1:
            conn = conn[:-1]
        elm_verts = mesh.vert[conn,:][:,:2]
        draw_polygon(elm_verts, ax, color=color, *args, **kwargs)

        if draw_numbers:
            mx, my = np.mean(elm_verts, axis=0)
            ax.annotate(i, xy=(mx,my))

    if scatter:
        ax.scatter(mesh.vert[:,0], mesh.vert[:,1], c=color)

    return ax

################################################################################
# Convenience methods
"""
Utilities for:
    labeling axes
    scaling figures
"""
################################################################################
def label_plot(ax, title=None, xlabel=None, ylabel=None):
    """ convenience method to annotate plot in one line
    @param ax  matplotlib axis object
    @param title  str title for plot
    @param xlabel  str xlabel
    @param ylabel  str ylabel
    """
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)

def colorbar(mappable):
    """ convenience method for adding colorbars to plots
    use:
    fig, ax = plt.subplots(1, 2)
    plot1 = ax[0].imshow(data)
    colorbar(plot1)
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%', pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def save_plot(fn, **kwargs):
    """ convenience method to save current figure
    @param fn  filename for the figure
    """
    plt.savefig(fn, bbox_inches='tight', **kwargs)
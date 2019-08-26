import numpy as np
import renderapi
import argschema
from ..schemas import *
from ..utils import make_dbconnection
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch


def fixpi(arr):
    # make the angular values fall around zero
    ind = np.argwhere(arr > np.pi)
    while len(ind) != 0:
        arr[ind] = arr[ind] - 2.0 * np.pi
        ind = np.argwhere(arr > np.pi)
    ind = np.argwhere(arr < -np.pi)
    while len(ind) != 0:
        arr[ind] = arr[ind] + 2.0 * np.pi
        ind = np.argwhere(arr < -np.pi)
    return arr


def make_patch(tile):
    pts = []
    pts.append((tile.minX, tile.minY))
    pts.append((tile.maxX, tile.minY))
    pts.append((tile.maxX, tile.maxY))
    pts.append((tile.minX, tile.maxY))
    pts.append((tile.minX, tile.minY))
    return PolygonPatch(Polygon(pts))


def make_transform_patches(tilespecs):
    # getting tilespecs ready for plotting
    patches = []
    shearlist = []
    xscalelist = []
    yscalelist = []
    rotlist = []
    xmin = 1e9
    xmax = -1e9
    ymin = 1e9
    ymax = -1e9
    border = 4000
    for ts in tilespecs:
        patches.append(make_patch(ts))
        shearlist.append(ts.tforms[-1].shear)
        xscalelist.append(ts.tforms[-1].scale[0])
        yscalelist.append(ts.tforms[-1].scale[1])
        rotlist.append(ts.tforms[-1].rotation)
        if ts.minX < xmin:
            xmin = ts.minX
        if ts.minY < ymin:
            ymin = ts.minY
        if ts.maxX > xmax:
            xmax = ts.maxX
        if ts.maxY > ymax:
            ymax = ts.maxY

    xscalelist = np.array(xscalelist)
    yscalelist = np.array(yscalelist)
    shearlist = fixpi(np.array(shearlist))
    rotlist = fixpi(np.array(rotlist))
    return (
            [patches, shearlist, rotlist, xscalelist, yscalelist],
            (xmin - border, xmax + border),
            (ymin - border, ymax + border))


class CheckTransforms(argschema.ArgSchemaParser):
    default_schema = BigFetaPlotSchema

    def run(self):
        self.make_plot(
                self.args['z1'],
                self.args['output_stack'],
                plot=self.args['plot'])

    def make_transform_plot(
            self,
            fig,
            i,
            j,
            k,
            xlim,
            ylim,
            patches,
            value,
            bar=True):
        # plot a map of the transform value
        cmap = plt.cm.plasma_r
        ax = fig.add_subplot(i, j, k)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        LC = PatchCollection(patches, cmap=cmap)
        LC.set_array(value)
        LC.set_edgecolor('none')
        ax.add_collection(LC)
        if bar:
            fig.colorbar(LC)
        ax.set_aspect('equal')
        ax.patch.set_color([0.5, 0.5, 0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def make_plot(self, z1, stack, thr=None, plot=True):
        stack['db_interface'] = 'render'
        stack_dbconnection = make_dbconnection(stack)

        tspecs = renderapi.tilespec.get_tile_specs_from_z(
                stack['name'][0],
                int(float(z1)),
                render=stack_dbconnection)
        tids = []
        tforms = []
        for ts in tspecs:
            tids.append(ts.tileId)
            tforms.append(ts.tforms[-1])

        fig = plt.figure(1, figsize=(16, 4))

        tpatches, xlim, ylim = make_transform_patches(tspecs)
        self.shear = tpatches[1]
        self.rotation = tpatches[2]
        self.xscale = tpatches[3]
        self.yscale = tpatches[4]
        print('z=%d in stack %s__%s__%s' % (
            z1,
            stack['owner'],
            stack['project'],
            stack['name'][0]))
        print("average shear: %0.2f" % tpatches[1].mean())
        print("average rotation: %0.2f" % tpatches[2].mean())
        print("average xscale: %0.2f" % tpatches[3].mean())
        print("average yscale: %0.2f" % tpatches[4].mean())

        if plot:
            axs = []
            for j in np.arange(1, 5):
                axs.append(self.make_transform_plot(
                    fig,
                    2,
                    2,
                    j,
                    xlim,
                    ylim,
                    tpatches[0],
                    tpatches[j]))
            axs[0].set_title('shear')
            axs[1].set_title('rotation')
            axs[2].set_title('xscale')
            axs[3].set_title('yscale')

            fname = '%s/transforms_%s_%d.pdf' % (
                    self.args['plot_dir'],
                    stack['name'][0],
                    z1)
            pdf = PdfPages(fname)
            pdf.savefig(fig)
            pdf.close()
            plt.ion()
            plt.show()
            print('wrote %s' % fname)
            self.outputname = fname


if __name__ == '__main__':
    mod = CheckTransforms(schema_type=BigFetaPlotSchema)
    mod.run()

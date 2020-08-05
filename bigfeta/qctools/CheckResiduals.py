from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import numpy as np
import renderapi
import argschema
from ..schemas import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import mpl_scatter_density


def transform_pq(tspecs, matches):
    p = []
    q = []
    p_transf = []
    q_transf = []
    tids = []
    for ts in tspecs:
        tids.append(np.array([x.tileId for x in ts]))
    tsp_ind0 = 0
    tsp_ind1 = 1
    if len(tspecs) == 1:
        tsp_ind1 = 0

    for match in matches:
        sind0 = np.argwhere(tids[tsp_ind0] == match['pId'])
        sind1 = np.argwhere(tids[tsp_ind1] == match['qId'])
        if (len(sind0) > 0) & (len(sind1) > 0):
            sind0 = sind0[0][0]
            sind1 = sind1[0][0]
            ip = np.flipud(np.rot90(np.array(match['matches']['p'])))
            iq = np.flipud(np.rot90(np.array(match['matches']['q'])))

            ip_transf = tspecs[tsp_ind0][sind0].tforms[-1].tform(ip)
            iq_transf = tspecs[tsp_ind1][sind1].tforms[-1].tform(iq)

            p.append(ip)
            q.append(iq)
            p_transf.append(ip_transf)
            q_transf.append(iq_transf)

    return [p, q, p_transf, q_transf]


class CheckResiduals(argschema.ArgSchemaParser):
    default_schema = BigFetaPlotSchema

    def run(self):
        self.compute_residuals()
        self.cmap = plt.cm.plasma_r
        if self.args['plot']:
            self.make_plots()

    def compute_residuals(self):
        stack_dbconnection = renderapi.connect(**self.args['output_stack'])
        match_dbconnection = renderapi.connect(**self.args['pointmatch'])

        tspecs = [
                renderapi.tilespec.get_tile_specs_from_z(
                    self.args['output_stack']['name'][0],
                    float(self.args['z1']),
                    render=stack_dbconnection)]
        if self.args['z1'] != self.args['z2']:
            tspecs.append(
                    renderapi.tilespec.get_tile_specs_from_z(
                        self.args['output_stack']['name'][0],
                        float(self.args['z2']),
                        render=stack_dbconnection))
        sectionIds = []
        for t in tspecs:
            sectionIds.append(t[0].layout.sectionId)
        if len(sectionIds) == 1:
            sectionIds.append(sectionIds[0])

        if self.args['zoff'] != 0:
            tmp = []
            for sid in sectionIds:
                try:
                    tmp.append(str(int(sid) + self.args['zoff']))
                except ValueError:
                    tmp.append(sid)
            sectionIds = tmp

        matches = renderapi.pointmatch.get_matches_from_group_to_group(
                self.args['pointmatch']['name'][0],
                sectionIds[0],
                sectionIds[1],
                render=match_dbconnection)

        self.p, self.q, self.p_transf, self.q_transf = \
            transform_pq(tspecs, matches)
        self.xy_ave = [
                0.5 * (p + q)
                for p, q in zip(self.p_transf, self.q_transf)]
        self.xy_diff = [
                (p - q)
                for p, q in zip(self.p_transf, self.q_transf)]
        self.rss = [
                np.sqrt(
                    np.power(x[:, 0], 2.0) +
                    np.power(x[:, 1], 2.0))
                for x in self.xy_diff]

        self.mx = ''
        self.my = ''
        self.mr = ''

        self.ident = 'owner: %s' % self.args['output_stack']['owner']
        self.ident += '\nproject: %s' % self.args['output_stack']['project']
        self.ident += '\nstack: %s' % self.args['output_stack']['name'][0]
        self.ident += '\n'+'collection: %s' % self.args['pointmatch']['name'][0]
        self.ident += '\n'+'z1,z2: %d,%d' % (self.args['z1'], self.args['z2'])

    def make_plot(
            self,
            ax,
            coord_choice='xya',
            color_choice='rss',
            colorbar=False,
            projection=None):

        sign = 1
        if coord_choice == 'xya':
            plot_coords = np.concatenate(self.xy_ave)
        elif coord_choice == 'p':
            plot_coords = np.concatenate(self.p)
        elif coord_choice == 'q':
            plot_coords = np.concatenate(self.q)
            sign = -1
        cmin = -self.args['threshold']
        cmax = self.args['threshold']
        if color_choice == 'x':
            c = sign * np.concatenate(self.xy_diff)[:, 0]
            xlab = self.mx
        elif color_choice == 'y':
            c = sign * np.concatenate(self.xy_diff)[:, 1]
            xlab = self.my
        elif color_choice == 'rss':
            c = np.concatenate(self.rss)
            xlab = self.mr
            cmin = 0

        if projection == 'scatter_density':
            density = ax.scatter_density(
                    plot_coords[:, 0],
                    plot_coords[:, 1],
                    c=c,
                    norm=Normalize(),
                    cmap=self.cmap)
        else:
            density = ax.scatter(
                    plot_coords[:, 0],
                    plot_coords[:, 1],
                    c=c,
                    cmap=self.cmap,
                    edgecolors=None)

        ax.set_aspect('equal')
        ax.patch.set_color([0.5, 0.5, 0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlabel(xlab)
        fig = plt.gcf()
        if colorbar:
            fig.colorbar(density)

        if self.args['threshold'] is not None:
            density.set_clim(cmin, cmax)

    def make_lc_plots(self, fig, projection=None):
        fig.clf()
        ax1 = fig.add_subplot(141, projection=projection)
        self.make_plot(
                ax1,
                coord_choice='p',
                color_choice='x',
                colorbar=True,
                projection=projection)
        ax1.set_title(r'$\Delta x$ as p')

        ax2 = fig.add_subplot(142, projection=projection)
        self.make_plot(
                ax2,
                coord_choice='q',
                color_choice='x',
                colorbar=True,
                projection=projection)
        ax2.set_title(r'$\Delta x$ as q')

        ax3 = fig.add_subplot(143, projection=projection)
        self.make_plot(
                ax3,
                coord_choice='p',
                color_choice='y',
                colorbar=True,
                projection=projection)
        ax3.set_title(r'$\Delta y$ as p')

        ax4 = fig.add_subplot(144, projection=projection)
        self.make_plot(
                ax4,
                coord_choice='q',
                color_choice='y',
                colorbar=True,
                projection=projection)
        ax4.set_title(r'$\Delta y$ as q')

    def make_plots(self):
        fig = plt.figure(1, figsize=(12, 7.5))
        fig.clf()

        ax1 = fig.add_subplot(131, projection='scatter_density')
        self.make_plot(
                ax1,
                coord_choice='xya',
                color_choice='x',
                projection='scatter_density')
        ax2 = fig.add_subplot(132, projection='scatter_density')
        self.make_plot(
                ax2,
                coord_choice='xya',
                color_choice='y',
                projection='scatter_density')
        ax3 = fig.add_subplot(133, projection='scatter_density')
        self.make_plot(
                ax3,
                coord_choice='xya',
                color_choice='rss',
                projection='scatter_density')

        ax1.set_title(self.ident + r'\n$\Delta x$', fontsize=10)
        ax2.set_title(self.ident + r'\n$\Delta y$', fontsize=10)
        ax3.set_title(
                self.ident +
                r'\n$\sqrt{\Delta x^2+\Delta y^2}$',
                fontsize=10)

        ax1.set_xlabel(self.mx, fontsize=12)
        ax2.set_xlabel(self.my, fontsize=12)
        ax3.set_xlabel(self.mr, fontsize=12)

        if self.args['savefig']:
            self.outputname = '%s/residuals_%s_%s_%d_%d.pdf' % (
                    self.args['plot_dir'],
                    self.args['output_stack']['name'][0],
                    self.args['pointmatch']['name'][0],
                    self.args['z1'],
                    self.args['z2'])
            pdf = PdfPages(self.outputname)
            pdf.savefig(fig, dpi=100)
            pdf.close()
            plt.ion()
            plt.show()
            print('wrote %s' % self.outputname)


if __name__ == '__main__':
    mod = CheckResiduals(schema_type=BigFetaPlotSchema)
    mod.run()

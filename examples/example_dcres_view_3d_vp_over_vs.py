import numpy as np
import sys, glob, fnmatch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import argparse

try:
    import pickle5 as pickle
    # import cPickle as pickle
except ImportError:
    import pickle
try:
    from evodcinv import DispersionCurve, layered_model, LayeredModel, ThomsonHaskell, params2lay
except ImportError:
    sys.path.append("../")
    from evodcinv import DispersionCurve, layered_model, LayeredModel, ThomsonHaskell, params2lay


def make_plottable_model(thickness_in, model_in, zmax):
    model_plot = []
    z_plot = []
    thickness = 0
    i = 0
    while (thickness < zmax) & (i < len(model_in)):
        model_plot.extend([model_in[i], model_in[i]])
        z_plot.extend([thickness, thickness + thickness_in[i]])
        thickness = thickness + thickness_in[i]
        i += 1
    z_plot[-1] = zmax
    return z_plot, model_plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", help="Choose between phase velocity (default) or group velocity",
                        type=str, default="phase")
    parser.add_argument("output_name", help="Enter output file name",
                        type=str, default="output")
    parser.add_argument("input_global_curves", help="Enter output file name",
                        type=str, default="data")
    parser.add_argument("input_dispersion_dict_tomo", help="Enter path to dictionary containing global dispersion data",
                        type=str, default="data/disp_curves_tomo.pickle")
    args = parser.parse_args()
    dtype = args.dtype
    output_name = args.output_name
    global_data_dir = args.input_global_curves
    tomo_file = args.input_dispersion_dict_tomo

    # Parameters
    skip = 1
    zmax = 1500
    vmin_group = 200
    vmax_group = 1000
    # colorby = 'apost'
    colorby = 'misfit'
    # clim_mode = 'fixed'
    clim_mode = 'normalized'
    misfit_bounds = [0, 1]
    outdir = "output/" + output_name # Output directory
    fwd_modelling_file = outdir + "/fwd_modelling.pickle"
    mean_model_file = outdir + "/mean_model_misfit_max_1.txt"
    min_model_file = outdir + "/min_models_misfit_max_1.txt"
    max_model_file = outdir + "/max_models_misfit_max_1.txt"
    #best_model_file = outdir + "/best_model.pickle"
    best_model_file = outdir + "/best_model_misfit_max_1.txt"
    figdir = outdir + "/figures"  # Output figures directory
    cmap = "viridis_r"  # Colormap
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    # import global dispersion curves
    disp_param = list()
    filenames = os.listdir('%s/' % global_data_dir)
    for name in filenames:
        if '.txt' in name:
            path = '%s/%s' % (global_data_dir, name)
            if 'rayleigh' in name:
                wtype = 'rayleigh'
            elif 'love' in name:
                wtype = 'love'
            else:
                continue
            mode = [int(s) for s in list(name) if s.isdigit()]
            mode = int(str(mode)[1:-1])
            disp_param.append((path, wtype, mode))
    dcurves_global = []
    for param in disp_param:
        filename, wtype, mode = param
        faxis, disp, uncertainties = np.loadtxt(filename, unpack=True)
        if 'group' in filename:
            dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype='group')
        else:
            dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype='phase')
        dc.dtype = dtype
        dcurves_global.append(dc)

    # import tomography dispersion curves
    with open(fwd_modelling_file, 'rb') as f1:
        dict_forward_modelling = pickle.load(f1)
        apost_global_dcurves = dict_forward_modelling['misfit_table'][:, 0]
        apost_tomo_dcurves = dict_forward_modelling['misfit_table'][:, 1]
        forward_dcurves_global = dict_forward_modelling['forward_dcurves_global']
        forward_dcurves_tomo = dict_forward_modelling['forward_dcurves_tomo']
        faxis_global = dict_forward_modelling['faxis_global'] # todo: reactivate this line after new run of dcres
    # faxis_file = outdir + "/faxis_global.pickle"
    # with open(faxis_file, 'rb') as f1:
        # faxis_global = pickle.load(f1) # todo: remove this line after new run of dcres

    # read models & energy
    with open(outdir + '/models.pickle', 'rb') as f1:
        models = pickle.load(f1)
    with open(outdir + '/energy.pickle', 'rb') as f1:
        energy = pickle.load(f1)
    n_models = len(models)

    best_model = models[np.argmin(energy)]

    # compute apost and prepare the related colormap
    apost = np.exp(-0.5 * energy ** 2)
    if colorby == 'apost':
        color_vals = apost
    elif colorby == 'misfit':
        color_vals = energy
    #norm = Normalize(min(apost), max(apost))
    if clim_mode == 'absolute':
        if colorby == 'apost':
            norm = Normalize(np.exp(-0.5 * misfit_bounds[1]**2), np.exp(-0.5 * misfit_bounds[0]**2))
        elif colorby == 'misfit':
            norm = Normalize(misfit_bounds[0], misfit_bounds[1])
    elif clim_mode == 'normalized':
        if colorby == 'apost':
            norm = Normalize(np.exp(-0.5 * max(energy)**2), np.exp(-0.5 * min(energy)**2))
        elif colorby == 'misfit':
            norm = Normalize(min(energy), max(energy))
    smap = ScalarMappable(norm, cmap)
    smap.set_array([])

    # read forward modelled dispersion curves
    with open(tomo_file, 'rb') as f1:
        dispersion_dict_tomo = pickle.load(f1)
        n_layers = dispersion_dict_tomo['true_model'].shape[1]

    # Import mean, min and max models
    model_mean_plot = np.loadtxt(mean_model_file)
    model_min_plot = np.loadtxt(min_model_file)
    model_max_plot = np.loadtxt(max_model_file)
    best_model_plot = np.loadtxt(best_model_file)
    # z_plot = model_min_plot[0, :]
    model_mean_vals = model_mean_plot[1, :]
    model_min_vals = model_min_plot[1, :]
    model_max_vals = model_max_plot[1, :]
    model_best_vals = best_model_plot[1, :]
    # with open(best_model_file, 'rb') as f1:
        # best_model = pickle.load(f1)

    # Import best model
    # todo: with the new version of example dcres - read as text file
    mean_true_model = np.nanmax(dispersion_dict_tomo['true_model'], axis=0)
    mean_true_thickness = mean_true_model[:, -1]
    # best_model_vals = best_model.model_vp_over_vs
    # _, model_best_vals = make_plottable_model(mean_true_thickness, best_model_vals, zmax=zmax)

    # Plot all models colored by apost relative to different misfits
    fig1 = plt.figure()
    fig1.patch.set_alpha(0.)
    ax1 = fig1.add_subplot(1, 1, 1)
    for i in range(n_models):
        z_plot, model_plot = make_plottable_model(mean_true_thickness, models[i, :], zmax=zmax)
        ax1.plot(model_plot, z_plot, color=smap.to_rgba(color_vals[i]))
    h_mean, = ax1.plot(model_mean_vals, z_plot, color='k', lw=2)
    h_best, = ax1.plot(model_best_vals, z_plot, color='r', ls='--', lw=2)
    ax1.set_ylim(z_plot[0], z_plot[-1])
    ax1.invert_yaxis()
    ax1.set_xlabel("Vp/Vs ratio", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.grid(True, linestyle=":")
    cb1 = fig1.colorbar(smap)
    cb1.set_label(colorby, fontsize=12)
    ax1.legend([h_mean, h_best], ['weighted average model', 'best model'],
               bbox_to_anchor=(1.3, 1), loc='upper left')
    plt.tight_layout()
    pos_ax1 = ax1.get_position()
    plt.savefig("%s/models_with_%s.png" % (figdir, colorby))
    plt.close(fig1)

    # Plot model envelopes
    fig1 = plt.figure()
    fig1.patch.set_alpha(0.)
    ax1 = fig1.add_subplot(1, 1, 1)
    h_area = ax1.fill_betweenx(z_plot, model_min_vals, model_max_vals, color='gray', alpha=0.5)
    h_mean, = ax1.plot(model_mean_vals, z_plot, color='k')
    h_best, = ax1.plot(model_best_vals, z_plot, color='r', ls='--')
    ax1.set_ylim(z_plot[0], z_plot[-1])
    ax1.invert_yaxis()
    ax1.set_xlabel("Vp/Vs ratio", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.grid(True, linestyle=":")
    ax1.legend([h_area, h_mean, h_best], ['range of models', 'weighted average model', 'best model'],
               bbox_to_anchor=(1.3, 1), loc='upper left')
    plt.tight_layout()
    ax1.set_position(pos_ax1)
    plt.savefig("%s/models_with_envelopes.png" % figdir)
    plt.close(fig1)

    # detect how many wave types there are in global data
    modes_ray = []
    modes_love = []
    wtypes = []
    for dcurve in dcurves_global:
        if 'rayleigh' in dcurve.wtype:
            if 'rayleigh' not in wtypes:
                wtypes.append("rayleigh")
            if dcurve.mode not in modes_ray:
                modes_ray.append(dcurve.mode)
        if 'love' in dcurve.wtype:
            if 'love' not in wtypes:
                wtypes.append("love")
            if dcurve.mode not in modes_love:
                modes_love.append(dcurve.mode)
    
    # Plot the global dispersion curves colored by apost relative to different misfits
    # along with data [1 figure for Rayleigh, 1 for Love]
    # fig2 = plt.figure(figsize=(5 * len(wtypes), 5), facecolor="white")
    # ax2 = [fig2.add_subplot(1, len(wtypes), i + 1) for i, w in enumerate(wtypes)]
    fig2, ax2 = plt.subplots(1, len(wtypes), constrained_layout=True, figsize = (8.5, 4.8))
    # fwd modelled curves :
    for (dcurve_forward, a) in zip(forward_dcurves_global, color_vals):
        for wtype in wtypes:
            modes_avail = [key for key in dcurve_forward[wtype].keys()]
            for mode in modes_avail:
                dcurve_plot = dcurve_forward[wtype][mode][dtype]['values']
                if wtype == 'rayleigh':
                    ax = ax2[max(0, len(wtypes) - 2)]
                    ax.set_title("Rayleigh modes")
                elif wtype == 'love':
                    ax = ax2[max(0, len(wtypes) - 1)]
                    ax.set_title("Love modes")
                ax.plot(faxis_global, dcurve_plot, color=smap.to_rgba(a))

    ax2[0].set_ylabel(f"{dtype} velocity (m/s)", fontsize=12)
    for ax in ax2:
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_xlim(min(faxis_global), max(faxis_global))
        ax.grid(True, linestyle=":")
    cb1 = fig2.colorbar(smap)
    cb1.set_label(colorby, fontsize=12)

    # data
    h_data_list1 = []
    str_list1 = []
    h_data_list2 = []
    str_list2 = []
    color_list = ['k', 'r', 'g', 'b']
    i_color_1 = 0
    i_color_2 = 0
    for dcurve_data in dcurves_global:
        if dcurve_data.wtype == 'rayleigh':
            ax = ax2[max(0, len(wtypes) - 2)]
            color = color_list[i_color_1]
        elif dcurve_data.wtype == 'love':
            ax = ax2[max(0, len(wtypes) - 1)]
            color = color_list[i_color_2]
        h_data = ax.errorbar(dcurve_data.faxis, dcurve_data.dtype_velocity, yerr=dcurve_data.uncertainties,
                    fmt="o", elinewidth=1, capsize=1.5, ms=3, mew=0.3,
                    ecolor=color, mfc=color, mec=color,
                    zorder=10, alpha=0.7)
        if dcurve_data.wtype == 'rayleigh':
            i_color_1 += 1
            h_data_list1.append(h_data)
            str_list1.append('FK - Mode ' + str(dcurve_data.mode))
        elif dcurve_data.wtype == 'love':
            i_color_2 += 1
            h_data_list2.append(h_data)
            str_list2.append('FK - Mode ' + str(dcurve_data.mode))
    if len(h_data_list2) > 0:
        ax2[-1].legend(h_data_list2, str_list2, loc='upper right')
    if len(h_data_list1) > 0:
        ax2[0].legend(h_data_list1, str_list1, loc='upper right')
    fig2.savefig(figdir + f'/{dtype}_dispersion_curves.png', dpi=400)
    plt.close(fig2)

    # Plot the tomo dispersion curves for specified cells colored by apost along with data
    # loop on tomo cells
    x_tomo_all = dispersion_dict_tomo['X']
    y_tomo_all = dispersion_dict_tomo['Y']
    for i, (x, y) in enumerate(zip(x_tomo_all, y_tomo_all)):
        # get geographical coords, prepare file name for figure
        str_title = 'group_dispersion_curves_tomo_cell_X_' + str(int(x)) + '_Y_' + str(int(y))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # plot forward modelled curves
        for (dcurve_forward_list, a) in zip(forward_dcurves_tomo, color_vals):
            dcurve_plot = dcurve_forward_list[i]
            ax.set_title("Rayleigh - tomography - cell X = " + str(int(x)) + '; Y = ' + str(int(y)))
            ax.plot(faxis_global, dcurve_plot, color=smap.to_rgba(a))

        # plot data
        h_data = ax.errorbar(dispersion_dict_tomo['f_axis'], dispersion_dict_tomo['DispersionCurves'][i].dtype_velocity,
                    yerr=dispersion_dict_tomo['DispersionCurves'][i].uncertainties, color='k',
                             fmt="o", elinewidth=1, capsize=1.5, ms=3, mew=0.3,
                             zorder=10, alpha=0.7)
        ax.legend([h_data], ['Tomography - mode 0'], loc='upper right')
        cb1 = fig.colorbar(smap)
        cb1.set_label(colorby, fontsize=12)
        ax.set_ylabel(f"{dispersion_dict_tomo['DispersionCurves'][i].dtype} velocity (m/s)", fontsize=12)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_xlim(min(faxis_global), max(faxis_global))
        ax.set_ylim(vmin_group, vmax_group)
        ax.grid(True, linestyle=":")
        plt.tight_layout()
        fig.savefig(figdir + f'/{str_title}.png', dpi=400)
        plt.close(fig)

    # make figure
    # next step : add inset to indicate station position




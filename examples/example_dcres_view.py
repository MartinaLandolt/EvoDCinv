# -*- coding: utf-8 -*-

import numpy as np
import sys, glob, fnmatch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os
import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from evodcinv import params2vel
except ImportError:
    sys.path.append("../")
    from evodcinv import params2vel

def to_group_velocity(phase_velocity,faxis):
    """
    Convert phase velocity to group velocity by
    differentiating phase_velocity

    Only works if frequencies are evenly spaced.
    """

    omega = 2 * np.pi * faxis
    domega = omega[1] - omega[0]
    if not np.allclose(np.diff(omega), domega, rtol=10 ** -2):
        raise ValueError("""Frequencies not evenly spaced. 
               Could not convert from phase velocity to group velocity""")
    dphase_domega = np.gradient(phase_velocity, domega)
    # group_velocity = phase_velocity  + omega * dphase_domega
    group_velocity = phase_velocity / (1 - omega / phase_velocity * dphase_domega)

    return group_velocity


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", help="Choose between phase velocity (default) or group velocity",
                        type=str, default="phase")
    parser.add_argument("output_name", help="Enter output file name",
                        type=str, default="output")
    parser.add_argument("input_name", help="Enter output file name",
                        type=str, default="data")
    dtype = parser.parse_args().dtype
    output_name = parser.parse_args().output_name
    input_name = parser.parse_args().input_name

    # Parameters
    fmin, fmax = 0.1, 10.
    skip = 100
    zmax = 1500
    data_dir = input_name
    outdir = "output/" + output_name # Output directory
    figdir = outdir + "/figures"  # Output figures directory
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    cmap = "viridis_r"  # Colormap
    # wtypes = ["rayleigh", "love"]
    wtypes = []
    all_filenames = os.listdir('%s/' % data_dir)
    filenames = list()
    for name in all_filenames:
        if '.txt' in name:
            filenames.append(name)

    if (any('group' in filenames[i] for i in range(len(filenames)))) and (dtype == 'phase'):
        raise Exception('Impossible to invert group velocity in phase velocity mode')

    r_filenames = list()
    l_filenames = list()
    ind_ray = []
    ind_love = []
    for name in filenames:
        if 'rayleigh' in name:
            if 'rayleigh' not in wtypes:
                wtypes.append("rayleigh")
            r_filenames.append(name)
            ind = [int(s) for s in list(name) if s.isdigit()]
            if ind not in ind_ray:
                ind = int(str(ind)[1:-1])
                ind_ray.append(ind)
        if 'love' in name:
            if 'love' not in wtypes:
                wtypes.append("love")
            l_filenames.append(name)
            ind = [int(s) for s in list(name) if s.isdigit()]
            if ind not in ind_love:
                ind = int(str(ind)[1:-1])
                ind_love.append(ind)

    # Unpickle required files
    models = pickle.load(open("%s/models.pickle" % outdir, "rb"))[::skip]
    energy = pickle.load(open("%s/energy.pickle" % outdir, "rb"))[::skip]
    if "rayleigh" in wtypes:
        rcurves = pickle.load(open("%s/rcurves.pickle" % outdir, "rb"))[::skip]
        if dtype == "group":
            for dcurves, e in zip(rcurves, energy):
                for dcurve in dcurves:
                    dcurve.dtype = dtype
    if "love" in wtypes:
        lcurves = pickle.load(open("%s/lcurves.pickle" % outdir, "rb"))[::skip]
        if dtype == "group":
            for dcurves, e in zip(lcurves, energy):
                for dcurve in dcurves:
                    dcurve.dtype = dtype

    # Convert acceptable models to continuous velocity models
    vel, az = np.transpose([params2vel(m, zmax=zmax) for m in models], axes=[1, 0, 2])

    # Import true models
    trueV_file = np.loadtxt("%s/true_model.txt" % data_dir, unpack=True)
    trueVs = trueV_file[1, :]
    trueThickness = trueV_file[3, :]

    # Import mean, min and max models
    model_moy = np.loadtxt('%s/mean_model.txt' % outdir, unpack=True)
    model_min = np.loadtxt('%s/min_models.txt' % outdir, unpack=True)
    model_max = np.loadtxt('%s/max_models.txt' % outdir, unpack=True)

    #Import best model
    best_model = pickle.load(open("%s/best_model_rms.pickle" % outdir, "rb"))
    n = int(len(best_model) / 3)
    bestVs_ax = []
    for i in range(n):
        bestVs_ax.extend([best_model[i], best_model[i]])
    bestThickness_ax = []
    thickness = 0
    for i in range(n):
        bestThickness_ax.extend([thickness, thickness+best_model[n+i]])
        thickness = thickness+best_model[n+i]

    # Import data dispersion curves
    if "rayleigh" in wtypes:
        Real_Rayleigh =  []
        for filename in sorted(r_filenames):
            data_rayleigh = np.loadtxt(open(("%s/%s" % (data_dir,filename)), "rb"), unpack=True)
            if (dtype == 'group') & ('group' not in filename):
                data_rayleigh[1] = to_group_velocity(data_rayleigh[1], data_rayleigh[0])
            Real_Rayleigh.append(data_rayleigh)

    if "love" in wtypes:
        Real_Love = []
        for filename in sorted(l_filenames):
            data_love = np.loadtxt(open(("%s/%s" % (data_dir,filename)), "rb"), unpack=True)
            if (dtype == 'group') & ('group' not in filename):
                data_love[1] = to_group_velocity(data_love[1], data_love[0])
            Real_Love.append(data_love)


    # Initialize figures
    fig1 = plt.figure(figsize=(5, 5), facecolor="white")
    fig1_bis = plt.figure(figsize=(5, 5), facecolor="white")
    fig2 = plt.figure(figsize=(5 * len(wtypes), 5), facecolor="white")

    fig1.patch.set_alpha(0.)
    fig1_bis.patch.set_alpha(0.)
    fig2.patch.set_alpha(0.)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1_bis = fig1_bis.add_subplot(1, 1, 1)
    ax2 = [fig2.add_subplot(1, len(wtypes), i + 1) for i, w in enumerate(wtypes)]

    # Make colormap
    apost = np.exp(-0.5*energy**2)
    norm = Normalize(min(apost), max(apost))
    #norm = Normalize(energy.min(), energy.max())
    smap = ScalarMappable(norm, cmap)
    smap.set_array([])

    # Plot velocity models
    for v, a, e in zip(vel, az, apost):
        ax1.plot(v, a, color=smap.to_rgba(e))

    # Plot true velocity model on top
    trueVs_ax = []
    for i in range(len(trueVs)):
        trueVs_ax.extend([trueVs[i], trueVs[i]])
    trueThickness_ax = []
    thickness = 0
    for i in range(len(trueVs)):
        trueThickness_ax.extend([thickness, thickness+trueThickness[i]])
        thickness = thickness+trueThickness[i]
    ax1.plot(trueVs_ax, trueThickness_ax, color='red', label='true model')

    # Plot mean, min and max model on top
    ax1.plot(model_moy[:, 1], model_moy[:, 0], color='magenta', label='mean model')
    ax1.fill_betweenx(model_min[:, 0], model_min[:, 1], model_max[:, 1], alpha=0.2)

    ax1.legend(loc='upper right')
    ax1.set_xlabel("Velocity (m/s)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.set_ylim(a[0], a[-1])
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":")

    cb1 = fig1.colorbar(smap)
    cb1.set_label("apost", fontsize=12)

    # Figure 1 bis: models only mean, max, min, best and true model represented
    ax1_bis.plot(trueVs_ax, trueThickness_ax, color='red', label='true model')
    ax1_bis.plot(model_moy[:, 1], model_moy[:, 0], color='magenta', label='mean model')
    ax1_bis.plot(bestVs_ax,bestThickness_ax, color='blue', label='best model')
    ax1_bis.fill_betweenx(model_min[:, 0], model_min[:, 1], model_max[:, 1], alpha=0.2, label='range of models')
    ax1_bis.legend(loc='upper right')
    ax1_bis.set_xlabel("Velocity (m/s)", fontsize=12)
    ax1_bis.set_ylabel("Depth (m)", fontsize=12)
    ax1_bis.set_ylim(a[0], a[-1])
    ax1_bis.invert_yaxis()
    ax1_bis.grid(True, linestyle=":")

    # PLot dispersion curves
    if "rayleigh" in wtypes:
        ax = ax2[max(0, len(wtypes) - 2)]
        for dcurves, e in zip(rcurves, apost):
            for dcurve in dcurves:
                dcurve.plot(axes=ax, plt_kws=dict(color=smap.to_rgba(e)))
        for mode in ind_ray:
            for freq, v in zip(Real_Rayleigh[mode][0], Real_Rayleigh[mode][1]):
                ax.scatter(freq, v, s=10, marker="+", facecolor="black", zorder=10, label="Real data")
        ax.set_title("Rayleigh-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel(f"{dtype} velocity (m/s)", fontsize=12)
        ax.set_xlim(fmin, fmax)
        ax.grid(True, linestyle=":")

    if "love" in wtypes:
        ax = ax2[max(0, len(wtypes) - 1)]
        for dcurves, e in zip(lcurves, apost):
            for dcurve in dcurves:
                dcurve.plot(axes=ax, plt_kws=dict(color=smap.to_rgba(e)))
        for mode in ind_love:
            for freq, v in zip(Real_Love[mode][0], Real_Love[mode][1]):
                ax.scatter(freq, v, s=10, marker="+", facecolor="black", zorder=10, label="Real data")
        ax.set_title("Love-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel(f"{dtype} velocity (m/s)", fontsize=12)
        ax.set_xlim(fmin, fmax)
        ax.grid(True, linestyle=":")

    fig1.tight_layout()
    fig1_bis.tight_layout()
    fig2.tight_layout()
    # fig1.show()
    # fig2.show()

    fig1.savefig(figdir + '/models_vitesse.png', dpi=400)
    fig1_bis.savefig(figdir + '/models_vitesse_moyen.png', dpi=400)
    fig2.savefig(figdir + f'/{dtype}_dispertion_curves.png', dpi=400)

    #figure 2 from Simon's script with uncertainties

    fig3 = plt.figure(figsize=(5 * len(wtypes), 5), facecolor="white")
    fig3.patch.set_alpha(0.)
    ax3 = [fig3.add_subplot(1, len(wtypes), i + 1) for i, w in enumerate(wtypes)]
    # Second Figure : Dispersion curves
    ind = np.argmin(energy)
    if "rayleigh" in wtypes:
        ax = ax3[max(0, len(wtypes) - 2)]
        for dcurves, e in zip(rcurves, apost):
            for dcurve in dcurves:
                dcurve.plot(axes=ax, plt_kws=dict(color=smap.to_rgba(e)))
        for mode in ind_ray:
            for freq, v, inc in zip(Real_Rayleigh[mode][0], Real_Rayleigh[mode][1], Real_Rayleigh[mode][2]):
                ax.errorbar(freq, v, yerr=inc, fmt="o", ecolor="red", elinewidth=1, capsize=1.5, mfc='r', mec='r', ms=3, mew=0.3, zorder=10, alpha=0.3)
        # for filename in sorted(r_filenames):
        #     R = np.loadtxt(open(("%s/%s" % (data_dir,filename)), "rb"), unpack=True)
        #     ax.errorbar(R[0], R[1], yerr=R[2], fmt="o", ecolor="blue", capsize=1.5, mfc='k', mec='k', ms=3, mew=1,
        #                 zorder=10)
        ax.set_title("Rayleigh-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Phase velocity (m/s)", fontsize=12)
        ax.set_xlim(fmin, fmax)
        #ax.set_ylim(500, 5000)
        ax.grid(True, linestyle=":")

    if "love" in wtypes:
        ax = ax3[max(0, len(wtypes) - 1)]
        for dcurves, e in zip(lcurves, apost):
            for dcurve in dcurves:
                dcurve.plot(axes=ax, plt_kws=dict(color=smap.to_rgba(e)))
        for mode in ind_love:
            for freq, v, inc in zip(Real_Love[mode][0], Real_Love[mode][1], Real_Love[mode][2]):
                ax.errorbar(freq, v, yerr=inc, fmt="o", ecolor="red", elinewidth=1, capsize=1.5, mfc='r', mec='r', ms=3, mew=0.3, zorder=10, alpha=0.3)
        # for filename in sorted(l_filenames):
        #     L = np.loadtxt(open(("%s/%s" % (data_dir,filename)), "rb"), unpack=True)
        #     ax.errorbar(L[0], L[1], yerr=L[2], fmt="o", ecolor="blue", capsize=1.5, mfc='k', mec='k', ms=3, mew=1,
        #                 zorder=10)
        ax.set_title("Love-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Phase velocity (m/s)", fontsize=12)
        ax.set_xlim(fmin, fmax)
        #ax.set_ylim(500, 5000)
        ax.grid(True, linestyle=":")

    fig3.savefig(figdir + '/dispertion_curves_uncertainties.png', dpi=400)

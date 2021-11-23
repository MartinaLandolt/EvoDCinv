# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

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

    dtype = 'group' #'phase'
    data_dir = 'data/groupV_model'
    outdir = "output/" + 'plot_synthetic_data_group'  # Output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    figdir = outdir + "/figures"  # Output figures directory
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    cmap = "viridis_r"  # Colormap
    fmin, fmax = 0.1, 10.
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

    # Import data dispersion curves
    if "rayleigh" in wtypes:
        Real_Rayleigh= []
        for filename in sorted(r_filenames):
            data_rayleigh = np.loadtxt(open(("%s/%s" % (data_dir, filename)), "rb"), unpack=True)
            if (dtype == 'group') & ('group' not in filename):
                data_rayleigh[1] = to_group_velocity(data_rayleigh[1], data_rayleigh[0])
            Real_Rayleigh.append(data_rayleigh)

    if "love" in wtypes:
        Real_Love = []
        for filename in sorted(l_filenames):
            data_love = np.loadtxt(open(("%s/%s" % (data_dir, filename)), "rb"), unpack=True)
            if (dtype == 'group') & ('group' not in filename):
                data_love[1] = to_group_velocity(data_love[1], data_love[0])
            Real_Love.append(data_love)

    # Import true models
    trueV_file = np.loadtxt("%s/true_model.txt" % data_dir, unpack=True)
    trueVs = trueV_file[1, :]
    trueThickness = trueV_file[3, :]


    # Initialize figures
    fig1 = plt.figure(figsize=(5, 5), facecolor="white")
    fig2 = plt.figure(figsize=(5 * len(wtypes), 5), facecolor="white")
    fig1.patch.set_alpha(0.)
    fig2.patch.set_alpha(0.)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = [fig2.add_subplot(1, len(wtypes), i + 1) for i, w in enumerate(wtypes)]


    # Plot true velocity model
    trueVs_ax = []
    for i in range(len(trueVs)):
        trueVs_ax.extend([trueVs[i], trueVs[i]])
    trueThickness_ax = []
    thickness = 0
    for i in range(len(trueVs)):
        trueThickness_ax.extend([thickness, thickness+trueThickness[i]])
        thickness = thickness+trueThickness[i]
    ax1.plot(trueVs_ax, trueThickness_ax, color='blue', label='true model')
    ax1.set_xlabel("Velocity (m/s)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.set_ylim(0, 1500)
    ax1.set_xlim(400, 1500)
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":")
    ax1.set_title("Synthetic velocity model")

    # PLot dispersion curves
    if "rayleigh" in wtypes:
        ax = ax2[max(0, len(wtypes) - 2)]
        for mode in ind_ray:
            for freq, v, inc in zip(Real_Rayleigh[mode][0], Real_Rayleigh[mode][1], Real_Rayleigh[mode][2]):
                ax.errorbar(freq, v, yerr=inc, fmt="o", ecolor="blue", capsize=1.5, mfc='k', mec='k', ms=3, mew=0.3, zorder=10, alpha=0.5)
            ax.plot(Real_Rayleigh[mode][0], Real_Rayleigh[mode][1], label='mode %i' %mode)
        ax.set_title("Rayleigh-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("%s velocity (m/s)" %dtype, fontsize=12)
        ax.legend()
        ax.set_xlim(fmin, fmax)
        ax.grid(True, linestyle=":")

    if "love" in wtypes:
        ax = ax2[max(0, len(wtypes) - 1)]
        for mode in ind_love:
            for freq, v, inc in zip(Real_Love[mode][0], Real_Love[mode][1], Real_Love[mode][2]):
                ax.errorbar(freq, v, yerr=inc, fmt="o", ecolor="blue", capsize=1.5, mfc='k', mec='k', ms=3, mew=0.3, zorder=10, alpha=0.5)
            ax.plot(Real_Love[mode][0], Real_Love[mode][1], label='mode %i' % mode)
        ax.set_title("Love-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("%s velocity (m/s)" %dtype, fontsize=12)
        ax.legend()
        ax.set_xlim(fmin, fmax)
        ax.grid(True, linestyle=":")

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(figdir + '/models_vitesse.png', dpi=400)
    fig2.savefig(figdir + '/dispertion_curves_uncertainties.png', dpi=400)



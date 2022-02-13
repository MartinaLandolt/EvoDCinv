# -*- coding: utf-8 -*-

import numpy as np
import sys, glob, os, fnmatch
from argparse import ArgumentParser
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy

try:
    from mpi4py import MPI
except ImportError:
    mpi_exist = False
else:
    mpi_exist = True

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
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_threads", type=int, default=8)
    parser.add_argument("-m", "--use_mpi", type=bool, default=True)
    parser.add_argument("dtype", help="Choose between phase velocity (default) or group velocity",
                        type=str, default="phase")
    parser.add_argument("output_name", help="Enter output file name",
                        type=str, default="output")
    parser.add_argument("input_global_curves", help="Enter output file name",
                        type=str, default="data")
    parser.add_argument("input_dispersion_dict_tomo", help="Enter path to dictionary containing global dispersion data",
                        type=str, default="data/disp_curves_tomo.pickle")
    args = parser.parse_args()
    num_threads = args.num_threads
    mpi = args.use_mpi
    dtype = args.dtype
    output_name = args.output_name
    global_data_dir = args.input_global_curves
    tomo_file = args.input_dispersion_dict_tomo

    if mpi and not mpi_exist:
        raise ValueError("mpi4py is not installed or not properly installed")

    if mpi:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
    else:
        mpi_rank = 0
        mpi_size = 1

    # Parameters
    # set parameters
    ny = 800
    perc = 1/np.sqrt(np.e)
    n_sigma_keep = 0.8
    outdir = "output/" + output_name
    zmin = 0
    zmax = 1500
    zstep = 5

    # load global dispersion curves
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

    # load tomography dispersion curves
    with open(tomo_file, 'rb') as f1:
        dispersion_dict_tomo = pickle.load(f1)
        n_layers = dispersion_dict_tomo['true_model'].shape[1]

    # Import inverted velocity models
    all_models, all_energy = [], []
    for filename in glob.glob("%s/run*.pickle" % outdir):
        m = pickle.load(open(filename, "rb"))
        all_models.append(np.hstack(m._models_vp_over_vs))
        all_energy.append(m.misfits.ravel())
    models = np.hstack(all_models).transpose()
    energy = np.hstack(all_energy)
    n_models = len(models)

    best_model = models[np.argmin(energy)]
    pickle.dump(best_model, open("%s/best_model_rms.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # best_model = []
    # best_model.append(pickle.load(open("%s/best_model_rms.pickle" % outdir, "rb")))

    # select good models
    apost = np.exp(-0.5*energy**2)
    # threshold = perc * apost.max()
    # idx = np.where(apost > threshold)[0]
    idx = np.where(energy < n_sigma_keep)[0]
    models = models[idx]
    energy = energy[idx]
    apost = apost[idx]

    # Keep good fitting models only : sort models
    idx = np.argsort(energy)[::-1]
    models = models[idx]
    energy = energy[idx]
    apost = apost[idx]

    # save good models
    pickle.dump(models, open("%s/models.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(energy, open("%s/energy.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # save the weighted average model and the min & max envelopes
    model_moy = np.empty(len(best_model))
    model_min = np.empty(len(best_model))
    model_max = np.empty(len(best_model))
    for z in range(len(best_model)):
        model_moy[z] = np.nansum([ (model[z]*np.exp(-0.5*energy[i]**2))
                                   for (i, model) in enumerate(models)])\
                       /np.nansum(np.exp(-0.5*energy**2))
        model_min[z] = np.nanmin([model[z] for model in models])
        model_max[z] = np.nanmax([model[z] for model in models])

    mean_true_model = np.nanmean(dispersion_dict_tomo['true_model'], axis=0)
    mean_true_thickness = mean_true_model[:, -1]
    z_plot, model_min_plot = make_plottable_model(mean_true_thickness, model_min, zmax=zmax)
    z_plot, model_max_plot = make_plottable_model(mean_true_thickness, model_max, zmax=zmax)
    z_plot, model_mean_plot = make_plottable_model(mean_true_thickness, model_moy, zmax=zmax)
    z_plot, model_best_plot = make_plottable_model(mean_true_thickness, best_model, zmax=zmax)
    fig = plt.figure()
    h_area = plt.gca().fill_betweenx(z_plot, model_min_plot, model_max_plot, color='gray', alpha=0.5)
    h_mean, = plt.plot(model_mean_plot, z_plot, color='k')
    h_best, = plt.plot(model_best_plot, z_plot, color='r', ls='--')
    plt.gca().invert_yaxis()
    plt.xlabel('Vp/Vs ratio')
    plt.ylabel('Depth (m)')
    plt.grid()
    plt.legend([h_area, h_mean, h_best], ['range of models', 'weighted average model', 'best model'],
               bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("%s/selected_models%s.png" % (outdir, '_misfit_max_' + str(n_sigma_keep)))
    plt.close(fig)

    # save plottable models :
    np.savetxt('%s/mean_model%s.txt' % (outdir, '_misfit_max_' + str(n_sigma_keep)), np.vstack([z_plot, model_mean_plot]))
    np.savetxt('%s/min_models%s.txt' % (outdir, '_misfit_max_' + str(n_sigma_keep)), np.vstack([z_plot, model_min_plot]))
    np.savetxt('%s/max_models%s.txt' % (outdir, '_misfit_max_' + str(n_sigma_keep)), np.vstack([z_plot, model_max_plot]))

    # initialize frequency axis characteristics (large and dense enough to contain all available modes)
    fmin = np.inf
    fmax = 0
    df = np.inf
    # prepare all necessary fields for forward modelling tomo data
    all_curves = dispersion_dict_tomo['DispersionCurves'] + dcurves_global
    for dcurve in all_curves:
        df_new = np.median(np.diff(dcurve.faxis))
        fmax_new = max(dcurve.faxis)
        fmin_new = min(dcurve.faxis)
        if df_new < df:
            df = df_new
        if fmin_new < fmin:
            fmin = fmin_new
        if fmax_new > fmax:
            fmax = fmax_new
    nf_ceil = int(np.ceil((fmax - fmin + df) / df))
    rest_ceil = int(np.ceil(nf_ceil / num_threads))
    nf_new = num_threads * rest_ceil
    faxis_global = np.linspace(fmin, fmax + df, nf_new)
    nf = len(faxis_global)

    # prepare container dictionary for average of forward modelling results
    modes_all = dict()
    for dcurve in all_curves:
        if dcurve.wtype not in modes_all.keys():
            modes_all[dcurve.wtype] = {}
        if dcurve.mode not in modes_all[dcurve.wtype].keys():
            modes_all[dcurve.wtype][dcurve.mode] = {}
        if dcurve.dtype not in modes_all[dcurve.wtype][dcurve.mode].keys():
            modes_all[dcurve.wtype][dcurve.mode][dcurve.dtype] = {'values': np.zeros(nf),
                                                                  'counts': np.zeros(nf)}

    # loop over good models
    n = len(models)

    if mpi:
        if mpi_rank==0:
            print('MPI mode verbose'+'\n')
            print('number of models to process' + '\n')
            print(n)
        mpi_comm.Barrier()
        mpi_comm.Bcast([models, MPI.DOUBLE], root=0)
        n_load = len(np.arange(mpi_rank, n, mpi_size))
        forward_dcurves_global = [None] * n_load
        forward_dcurves_tomo = [None] * n_load
        misfit_table = np.nan * np.zeros((n_load, 2))
        for (i, i_loc) in zip(np.arange(mpi_rank, n, mpi_size), range(n_load)):
            m = models[i]
            lm = LayeredModel()
            lm._dcurves = dcurves_global
            lm._n_layers = m.shape[0]
            lm._dispersion_dict_tomo = dispersion_dict_tomo
            lm._faxis_global = faxis_global
            lm._modes_all = copy.deepcopy(modes_all)
            # forward model via adapted cost function
            misfit_cumulated_new, misfit_global_new, misfit_tomo_new,  modes_all_new, dc_calc_all_new = \
                lm._costfunc_3d(m, ny, num_threads, 0.5)
            misfit_table[i_loc, 0] = misfit_global_new
            misfit_table[i_loc, 1] = misfit_tomo_new
            forward_dcurves_global[i_loc] = modes_all_new
            forward_dcurves_tomo[i_loc] = dc_calc_all_new
        mpi_comm.Barrier()
        mpi_comm.Allgather([misfit_table, MPI.DOUBLE],
                           [forward_dcurves_global, MPI.DOUBLE],
                           [forward_dcurves_tomo, MPI.DOUBLE])
        print('MPI loop finished' + '\n')
        print('len(misfit_table)'+'\n')
        print(len(misfit_table))
        print('len(forward_dcurves_global)'+'\n')
        print(len(forward_dcurves_global))
    else:
        forward_dcurves_global = []
        forward_dcurves_tomo = []
        misfit_table = np.nan * np.zeros((n, 2))
        for (i, m) in enumerate(models):
            lm = LayeredModel()
            lm._dcurves = dcurves_global
            lm._n_layers = m.shape[0]
            lm._dispersion_dict_tomo = dispersion_dict_tomo
            lm._faxis_global = faxis_global
            lm._modes_all = copy.deepcopy(modes_all)
            # forward model via adapted cost function
            misfit_cumulated_new, misfit_global_new, misfit_tomo_new,  modes_all_new, dc_calc_all_new = \
                lm._costfunc_3d(m, ny, num_threads, 0.5)
            misfit_table[i, 0] = misfit_global_new
            misfit_table[i, 1] = misfit_tomo_new
            forward_dcurves_global.append(modes_all_new)
            forward_dcurves_tomo.append(dc_calc_all_new)
            print('stop')

    # save the full dict per model (as in cost function)
    dict_forward_modelling = {'forward_dcurves_global': forward_dcurves_global,
                              'forward_dcurves_tomo': forward_dcurves_tomo,
                              'misfit_table': misfit_table}
    pickle.dump(dict_forward_modelling, open("%s/fwd_modelling.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


    # remains for dcres_view:
    # - plot the models using average width per layer, color by cost fun
    # - plot the fit to global data (standard, color by cost fun)
    # - plot the fit to individual tomo cells (specified by user as input, color by cost fun)



    # #fmin, fmax, df = 0.1, 10., 0.1
    # #f = np.arange(fmin, fmax+df, df)
    # fmin, fmax, nb_f = 0.1, 10., 100
    # f = np.linspace(fmin, fmax, nb_f)
    # ny = 800                          # Number of velocity discretization points
    # #perc = 90                         # Maximum RMS threshold as a percentage of best fitting models
    # perc = 1/np.sqrt(np.e)
    # n_sigma_keep = 1
    # zmin = 0; zmax = 2000; z_axis_step = 10
    # # data_dir = input_name
    # outdir = "output/" + output_name
    # figdir = outdir + "/figures"  # Output figures directory
    # if not os.path.exists(figdir):
    #     os.makedirs(figdir)
    # # Output directory
    # cmap = "viridis_r"                  # Colormap
    # all_filenames = os.listdir('%s/' % data_dir)
    # filenames = list()
    # for name in all_filenames:
    #     if '.txt' in name:
    #         filenames.append(name)
    # wtypes = []
    #
    # if (any('group' in filenames[i] for i in range(len(filenames)))) and (dtype=='phase'):
    #     raise Exception('Impossible to invert group velocity in phase velocity mode')
    #
    # r_filenames = list()
    # l_filenames = list()
    # ind_ray = []
    # ind_love = []
    # for name in filenames:
    #     if 'rayleigh' in name:
    #         if 'rayleigh' not in wtypes:
    #             wtypes.append("rayleigh")
    #         r_filenames.append(name)
    #         ind = [int(s) for s in list(name) if s.isdigit()]
    #         if ind not in ind_ray:
    #             ind = int(str(ind)[1:-1])
    #             ind_ray.append(ind)
    #     if 'love' in name:
    #         if 'love' not in wtypes:
    #             wtypes.append("love")
    #         l_filenames.append(name)
    #         ind = [int(s) for s in list(name) if s.isdigit()]
    #         if ind not in ind_love:
    #             ind = int(str(ind)[1:-1])
    #             ind_love.append(ind)
    #
    # # Import inverted velocity models
    # all_models, all_energy = [], []
    # for filename in glob.glob("%s/run*.pickle" % outdir):
    #     m = pickle.load(open(filename, "rb"))
    #     all_models.append(np.hstack(m.models))
    #     all_energy.append(m.misfits.ravel())
    # models = np.hstack(all_models).transpose()
    # energy = np.hstack(all_energy)
    # n_models = len(models)
    #
    # best_model = models[np.argmin(energy)]
    # pickle.dump(best_model, open("%s/best_model_rms.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    #
    # best_model = []
    # best_model.append(pickle.load(open("%s/best_model_rms.pickle" % outdir, "rb")))
    #
    # # Recompute dispersion curves of best_model
    # lay_best = np.array([params2lay(m) for m in best_model])
    #
    # if "rayleigh" in wtypes:
    #     rcurves_best = []
    #     for l in lay_best:
    #         th = ThomsonHaskell(l, wtype="rayleigh")
    #         th.propagate(f, ny=ny, domain="fc", n_threads=args.num_threads)
    #         dc_calculated = th.pick(modes=ind_ray)
    #         if dtype == "group":
    #             for dcurve in dc_calculated:
    #                 try:
    #                     dcurve.dtype = dtype
    #                 except:
    #                     dc_calculated = th.pick(modes=ind_ray)
    #         rcurves_best.append(dc_calculated)
    #     for nb_modes in ind_ray:
    #         filename = "%s/rayleigh_bestDC_mode%d.txt" % (outdir, nb_modes)
    #         rcurves_best[0][nb_modes].save(filename)
    #
    # if "love" in wtypes:
    #     lcurves_best = []
    #     for ind_love_model, l in enumerate(lay_best):
    #         th = ThomsonHaskell(l, wtype="love")
    #         th.propagate(f, ny=ny, domain="fc", n_threads=args.num_threads)
    #         dc_calculated = th.pick(modes=ind_love)
    #         if dtype == "group":
    #             for dcurve in dc_calculated:
    #                 try:
    #                     dcurve.dtype = dtype
    #                 except:
    #                     dc_calculated = th.pick(modes=ind_love)
    #         lcurves_best.append(dc_calculated)
    #     for nb_modes in ind_love:
    #         filename = "%s/love_bestDC_mode%d.txt" % (outdir, nb_modes)
    #         lcurves_best[0][nb_modes].save(filename)
    #
    #
    # # Keep good fitting models only : load data dc
    # if "rayleigh" in wtypes:
    #     Real_Rayleigh, Model_Rayleigh = [], []
    #     for filename in sorted(r_filenames):
    #         data_rayleigh = np.loadtxt(open(("%s/%s" % (data_dir,filename)), "rb"), unpack=True)
    #         if (dtype == 'group') & ('group' not in filename):
    #             data_rayleigh[1] = to_group_velocity(data_rayleigh[1], data_rayleigh[0])
    #         Real_Rayleigh.append(data_rayleigh)
    #     for filename in sorted(glob.glob("%s/rayleigh_bestDC_mode*.txt" % outdir)):
    #         data_rayleigh = np.loadtxt(open(filename, "rb"), unpack=True)
    #         if (dtype == 'group') & ('group' not in filename):
    #             data_rayleigh[1] = to_group_velocity(data_rayleigh[1], data_rayleigh[0])
    #         Model_Rayleigh.append(data_rayleigh) #This is best model dispersion curves saved above
    #
    # if "love" in wtypes:
    #     Real_Love, Model_Love = [], []
    #     for filename in sorted(l_filenames):
    #         data_love = np.loadtxt(open(("%s/%s" % (data_dir,filename)), "rb"), unpack=True)
    #         if (dtype == 'group') & ('group' not in filename):
    #             data_love[1] = to_group_velocity(data_love[1], data_love[0])
    #         Real_Love.append(data_love)
    #     for filename in sorted(glob.glob("%s/love_bestDC_mode*.txt" % outdir)):
    #         data_love = np.loadtxt(open(filename, "rb"), unpack=True)
    #         if (dtype == 'group') & ('group' not in filename):
    #             data_love[1] = to_group_velocity(data_love[1], data_love[0])
    #         Model_Love.append(data_love)
    #
    # # Keep good fitting models only : find differences modeled dc and data dc
    # def find_nearest(array, value):
    #     idx = (np.abs(array - value)).argmin()
    #     return array[idx], idx
    #
    # diff_vel = []
    # if "rayleigh" in wtypes:
    #     for nb_modes in ind_ray:
    #         for freq, v, u in zip(Real_Rayleigh[nb_modes][0],Real_Rayleigh[nb_modes][1], Real_Rayleigh[nb_modes][2]):
    #             value, indx = find_nearest(Model_Rayleigh[nb_modes][0],freq)
    #             diff_vel.append((v-Model_Rayleigh[nb_modes][1][indx])/u)
    # if "love" in wtypes:
    #     for nb_modes in ind_love:
    #         for freq, v, u in zip(Real_Love[nb_modes][0],Real_Love[nb_modes][1], Real_Love[nb_modes][2]):
    #             value, indx = find_nearest(Model_Love[nb_modes][0],freq)
    #             diff_vel.append((v-Model_Love[nb_modes][1][indx])/u)
    #
    # # Keep good fitting models only : calculate uncertainties
    # mu, std = norm.fit(diff_vel)
    # xmin, xmax = mu - 5 * std, mu + 5 * std
    # x = np.linspace(xmin, xmax, 100)
    # p = norm.pdf(x, mu, std)
    # value_minus1sig, indx_minus1sig = find_nearest(x, mu - 1 * std)
    # value_plus1sig, indx_plus1sig = find_nearest(x, mu + 1 * std)
    # inc_value = (abs(value_minus1sig) + abs(value_plus1sig)) / 2
    #
    # # Keep good fitting models only : choose models inside uncertainties bars
    # #energy=energy/inc_value
    # apost = np.exp(-0.5*energy**2)
    # threshold = perc * apost.max()
    # idx = np.where(apost > threshold)[0]
    # #idx = np.where(energy < n_sigma_keep)[0]
    # models = models[idx]
    # energy = energy[idx]
    # apost = apost[idx]
    #
    # # Keep good fitting models only : sort models
    # idx = np.argsort(energy)[::-1]
    # models = models[idx]
    # energy = energy[idx]
    # apost = apost[idx]
    #
    # pickle.dump(models, open("%s/models.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(energy, open("%s/energy.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # Keep good fitting models only : save mean, min and max of kept models
    # z_axis = np.arange(zmin, zmax + z_axis_step, z_axis_step)
    #     # discretisize model on this z axis
    #
    # def z_disctretisation(model, z_axis, n):
    #     vel_new = np.nan * np.zeros(len(z_axis))
    #     rho_new = np.nan * np.zeros(len(z_axis))
    #     z_cum = np.hstack((0, np.cumsum(model[n:2 * n])))
    #     for i in range(len(z_cum) - 1):
    #         z_min = z_cum[i]
    #         z_max = z_cum[i + 1]
    #         vel_new[(z_axis < z_max) & (z_axis >= z_min)] = model[i]
    #         rho_new[(z_axis < z_max) & (z_axis >= z_min)] = model[i+2*n]
    #     m_new = np.vstack([z_axis,vel_new, rho_new])
    #     return m_new
    #
    # n = int(len(models[0, :]) / 3)
    # new_models = []
    # for mod in models:
    #     m_new = z_disctretisation(mod, z_axis, n)
    #     new_models.append(m_new)
    #
    # model_moy = np.empty(len(z_axis))
    # model_min = np.empty(len(z_axis))
    # model_max = np.empty(len(z_axis))
    # for z in range(len(z_axis)):
    #     model_moy[z] = np.nansum([ (model[1,z]*np.exp(-0.5*energy[i]**2))
    #                                for (i,model) in enumerate(new_models)])\
    #                    /np.nansum(np.exp(-0.5*energy**2))
    #     model_min[z] = np.nanmin([model[1,z] for model in new_models])
    #     model_max[z] = np.nanmax([model[1,z] for model in new_models])
    #
    # np.savetxt('%s/mean_model.txt' % outdir, np.vstack([z_axis, model_moy]))
    # np.savetxt('%s/min_models.txt' % outdir, np.vstack([z_axis, model_min]))
    # np.savetxt('%s/max_models.txt' % outdir, np.vstack([z_axis, model_max]))
    #
    #
    # # Uncertainty figure chowing limit for models chosen
    #
    # fig1 = plt.figure(figsize=(5, 5), facecolor="white")
    # fig1.patch.set_alpha(0.)
    # ax = fig1.add_subplot(1, 1, 1)
    # ax.plot(x, p)
    # ax.hist(diff_vel)
    # plim = ax.get_ylim()
    # ax.plot(np.ones(len(plim)) * x[indx_minus1sig], [0, plim[1]], '--r', lw=2)
    # ax.plot(np.ones(len(plim)) * x[indx_plus1sig], [0, plim[1]], '--r', lw=2, label="Good fitting model limits")
    # plt.xlabel('Difference data and synthesized models')
    # plt.ylabel('number of synthesized models (normalized)')
    # plt.title('Incertitude histogram')
    # fig1.tight_layout()
    # fig1.savefig('%s/uncert.png' % figdir)
    #
    # # Figure showing data dc ( Real_Rayleigh) and Model_Rayleigh=?
    # fig2 = plt.figure(figsize=(5 * len(wtypes), 5), facecolor="white")
    # fig2.patch.set_alpha(0.)
    # ax2 = [fig2.add_subplot(1, len(wtypes), i + 1) for i, w in enumerate(wtypes)]
    #
    # if "rayleigh" in wtypes:
    #     ax = ax2[max(0, len(wtypes) - 2)]
    #     for nb_modes in ind_ray:
    #         for freq, v in zip(Real_Rayleigh[nb_modes][0], Real_Rayleigh[nb_modes][1]):
    #             ax.scatter(freq, v, s=10, marker="+", facecolor="black", zorder=10, label="Real data")
    #             ax.plot(Model_Rayleigh[nb_modes][0], Model_Rayleigh[nb_modes][1],
    #                     color=[0.993248, 0.906157, 0.143936, 1.], label="Modeled data")
    #             ax.set_title("Rayleigh-wave")
    #             ax.set_xlabel("Frequency (Hz)", fontsize=12)
    #             ax.set_ylabel("Phase velocity (m/s)", fontsize=12)
    #             ax.set_xlim(fmin, fmax)
    #             ax.grid(True, linestyle=":")
    #
    # if "love" in wtypes:
    #     ax = ax2[max(0, len(wtypes) - 1)]
    #     for nb_modes in ind_love:
    #         for freq, v in zip(Real_Love[nb_modes][0], Real_Love[nb_modes][1]):
    #             ax.scatter(freq, v, s=10, marker="+", facecolor="black", zorder=10)
    #             ax.plot(Model_Love[nb_modes][0], Model_Love[nb_modes][1], color=[0.993248, 0.906157, 0.143936, 1.])
    #             ax.set_title("Love-wave")
    #             ax.set_xlabel("Frequency (Hz)", fontsize=12)
    #             ax.set_ylabel("Phase velocity (m/s)", fontsize=12)
    #             ax.set_xlim(fmin, fmax)
    #             ax.grid(True, linestyle=":")
    #
    # fig2.tight_layout()
    # fig2.savefig('%s/working.png' % figdir)
    #
    # # Recompute dispersion curves
    # lay = np.array([params2lay(m) for m in models])
    # if "rayleigh" in wtypes:
    #     rcurves = []
    #     i = 0
    #     for l in lay:
    #         if i % 500 == 0:
    #             print("Rayleigh wave calculation : model ", i, " out of ", len(lay))
    #         i = i + 1
    #         # if i==462:
    #         #     print("error faxis=8,4 dosen't exist")
    #         th = ThomsonHaskell(l, wtype="rayleigh")
    #         th.propagate(f, ny=ny, domain="fc", n_threads=args.num_threads)
    #         dc_calculated = th.pick(modes=ind_ray)
    #         if dtype == "group":
    #             for dcurve in dc_calculated:
    #                 try:
    #                     dcurve.dtype = dtype
    #                 except:
    #                     dc_calculated = th.pick(modes=ind_ray)
    #                 # if i == 1593:
    #                 #     dc_calculated = th.pick(modes=ind_ray)
    #                 # if dcurve.flag_stop:
    #                 #     dc_calculated = th.pick(modes=ind_ray)
    #         rcurves.append(dc_calculated)
    #     pickle.dump(rcurves, open("%s/rcurves.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    #
    # if "love" in wtypes:
    #     i = 0
    #     lcurves = []
    #     for l in lay:
    #         flag_empty_mode = 0
    #         if i % 500 == 0:
    #             print("Love wave calculation : model ", i, " out of ", len(lay))
    #         i = i + 1
    #         th = ThomsonHaskell(l, wtype="love")
    #         th.propagate(f, ny=ny, domain="fc", n_threads=args.num_threads)
    #         dc_calculated = th.pick(modes=ind_love)
    #         for dcurve in dc_calculated:
    #             if len(dcurve.faxis) < 2:
    #                 flag_empty_mode = 1
    #         if flag_empty_mode == 0:
    #             if dtype == "group":
    #                 for dcurve in dc_calculated:
    #                     dcurve.dtype = dtype
    #             lcurves.append(dc_calculated)
    #     pickle.dump(lcurves, open("%s/lcurves.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    #
    # if "rayleigh" in wtypes:
    #     pickle.dump(rcurves, open("%s/rcurves.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # if "love" in wtypes:
    #     pickle.dump(lcurves, open("%s/lcurves.pickle" % outdir, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    #
    #
    # # Print output statistics
    # print("RMS min.: %.3f" % energy.min())
    # print("RMS %.1f%%: %.3f" % (perc, np.sqrt(-2. * np.log(threshold))))
    # print("Number of models kept: %d/%d (%.1f%%)" % (len(idx), n_models, len(idx) / n_models * 100))




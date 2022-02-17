# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import os, sys, time
from argparse import ArgumentParser
from copy import deepcopy
import pickle5 as pickle
try:
    from mpi4py import MPI
    mpi_exist = True
except ImportError:
    mpi_exist = False
try:
    from evodcinv import DispersionCurve, LayeredModel, progress, ThomsonHaskell
except ImportError:
    sys.path.append("../")
    from evodcinv import DispersionCurve, LayeredModel, progress, ThomsonHaskell


if __name__ == "__main__":
    # Initialize MPI
    if mpi_exist:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        print('mpi exists scenario, rank = ', str(mpi_rank))
    else:
        mpi_rank = 0
        print('mpi doesn t exist scenario')
        
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_threads", type=int, default=8)
    parser.add_argument("dtype", type=str, default="phase")
    parser.add_argument("output_name", help="Enter output file name",
                        type=str, default="output")
    parser.add_argument("input_global_curves", help="Enter path to folder containing global dispersion data",
                        type=str, default="data")
    parser.add_argument("input_dispersion_dict_tomo", help="Enter path to dictionary containing global dispersion data",
                        type=str, default="data/disp_curves_tomo.pickle")
    args = parser.parse_args()
    dtype = args.dtype
    output_name = parser.parse_args().output_name
    global_data_dir = parser.parse_args().input_global_curves
    tomo_file = parser.parse_args().input_dispersion_dict_tomo


    # Parameters
    ny = 800 #200                        # Number of velocity discretization points
    max_run = 10                    # Number of runs
    popsize = 16
    max_iter = 50
    outdir = "output/" + output_name             # Output directory


    # Import dispersion dict
    with open(tomo_file, 'rb') as f1:
        dispersion_dict_tomo = pickle.load(f1)

    # Count layers
    n_layers = dispersion_dict_tomo['true_model'].shape[1]

    # Inversion boundaries
    vp_over_vs = np.tile(np.array([1.5, 3.5]), (n_layers, 1))

    # Check the number of layers is consistent
    assert vp_over_vs.shape[0] == n_layers

    # print(vp_over_vs)
    
    # Initialize dispersion curves
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
    # disp_param = [
    #     ( "data/rayleigh_mode0.txt", "rayleigh", 0 ),
    #     ( "data/rayleigh_mode1.txt", "rayleigh", 1 ),
    #     ( "data/love_mode0.txt", "love", 0 ),
    #     ( "data/love_mode1.txt", "love", 1 ),
    #     ]

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

    # Evolutionary optimizer parameters
    evo_kws = dict(popsize=popsize, max_iter=max_iter, constrain=True, mpi=mpi_exist) #, random_state=1 to test same invertion on several runs
    opt_kws = dict(solver="cpso")

    # Multiple inversions
    if mpi_rank == 0:
        print("Run with population size = %i" % popsize)
        print("Maximum iteration = %i" % max_iter)
        print("Number of runs = %i" % max_run)
        print("Wave type of the data = %s" % dtype)
        print("Number of velocity discretization points, ny = %i" % ny)
        starttime = time.time()
        os.makedirs(outdir, exist_ok=True)
        progress(-1, max_run, "perc", prefix="Inverting dispersion curves: ")

    models = []
    for i in range(max_run):
        lm = LayeredModel()
        lm.invert3D_fixed_interfaces(dcurves_global, dispersion_dict_tomo, vp_over_vs, ny=ny, n_threads=args.num_threads,
                  weight_global_curves=0.5, evo_kws=evo_kws, opt_kws=opt_kws)
        # lm.invert(dcurves, beta, thickness, ny=ny, n_threads=args.num_threads,
        #           evo_kws=evo_kws, opt_kws=opt_kws)
        
        # test up to here, then see what to save and how for further visualization steps
        if mpi_rank == 0:
            lm.save("%s/run%d.pickle" % (outdir, i + 1))
            models.append(deepcopy(lm))
            progress(i, max_run, "perc", prefix="Inverting dispersion curves: ")

    if mpi_rank == 0:
        print("\n")
        #:todo adapt to size of m
        misfits = [m.misfit for m in models]
        print(models[np.argmin(misfits)]._model_vp_over_vs)
        print("Elapsed time: %.2f seconds\n" % (time.time() - starttime))

        #:todo adapt to size of m
        best_model = models[np.argmin(misfits)]
        best_model.save("%s/best_model.pickle" % outdir)

        #: todo create new routine params2lay adapted to the case with thickness and Vp fixed
        #vel = best_model.params2lay()
        #th = ThomsonHaskell(vel, dcurves_global[0].wtype)
        #th.propagate(dcurves_global[0].faxis, ny=ny, domain="fc")
        #inverted_dc = th.pick([dcurves_global[0].mode])[0]
        #inverted_dc.dtype = dcurves_global[0].dtype
        #inverted_dc.save("%s/best_model_%s_dc.txt" % (outdir, dtype))

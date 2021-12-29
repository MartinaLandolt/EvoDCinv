# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import os, sys, time
from argparse import ArgumentParser
from copy import deepcopy
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
    parser.add_argument("input_name", help="Enter output file name",
                        type=str, default="data")
    args = parser.parse_args()
    dtype = args.dtype
    output_name = parser.parse_args().output_name
    data_dir = parser.parse_args().input_name
        
    # Parameters
    ny = 800 #200                        # Number of velocity discretization points
    max_run = 10                    # Number of runs
    popsize = 20
    max_iter = 200
    outdir = "output/" + output_name             # Output directory

    # Inversion boundaries
    beta = np.array([[200., 1000.],
                      [1000., 3000.],
                      [1000., 3000.],
                      [1000., 3000.],
                      [1000., 3000.],
                      [1000., 3000.],
                      [1000., 3000.],
                      [1000., 3000.]])
    thickness = np.array([[5., 50.],
                           [50., 1000.],
                           [50., 1000.],
                           [50., 1000.],
                           [50., 1000.],
                           [50., 1000.],
                           [50., 1000.],
                           [99999., 99999.]])
    print(beta)
    
    # Initialize dispersion curves
    disp_param = list()
    filenames = os.listdir('%s/' % data_dir)
    for name in filenames:
        if '.txt' in name:
            path = '%s/%s' % (data_dir, name)
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

    dcurves = []
    for param in disp_param:
        filename, wtype, mode = param
        faxis, disp, uncertainties = np.loadtxt(filename, unpack=True)
        if 'group' in filename:
            dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype='group')
        else:
            dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype='phase')
        dc.dtype = dtype
        dcurves.append(dc)

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
        lm.invert(dcurves, beta, thickness, ny=ny, n_threads=args.num_threads,
                  evo_kws=evo_kws, opt_kws=opt_kws)
        if mpi_rank == 0:
            lm.save("%s/run%d.pickle" % (outdir, i + 1))
            models.append(deepcopy(lm))
            progress(i, max_run, "perc", prefix="Inverting dispersion curves: ")

    if mpi_rank == 0:
        print("\n")
        misfits = [m.misfit for m in models]
        print(models[np.argmin(misfits)])
        print("Elapsed time: %.2f seconds\n" % (time.time() - starttime))

        best_model = models[np.argmin(misfits)]
        best_model.save("%s/best_model.pickle" % outdir)
        vel = best_model.params2lay()
        th = ThomsonHaskell(vel, dcurves[0].wtype)
        th.propagate(dcurves[0].faxis, ny=ny, domain="fc")
        inverted_dc = th.pick([dcurves[0].mode])[0]
        inverted_dc.dtype = dcurves[0].dtype
        inverted_dc.save("%s/best_model_%s_dc.txt" % (outdir, dtype))

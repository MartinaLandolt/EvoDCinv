# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from stochopy import Evolutionary
from .dispersion_curve import DispersionCurve
from .thomson_haskell import ThomsonHaskell
from ._lay2vel import lay2vel as l2vf
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
__all__ = [ "LayeredModel", "params2lay", "params2vel" ]

    
class LayeredModel:
    """
    Layered velocity model
    
    This class inverts for a layered medium given different modes of observed
    dispersion curves.
    
    Parameters
    ----------
    model : ndarray
        Layered velocity model.
    """
    
    def __init__(self, model = None):
        if model is not None and not isinstance(model, np.ndarray) and model.ndim != 2:
            raise ValueError("model must be a 2-D ndarray")
        if model is not None and model.shape[1] != 4:
            raise ValueError("model must have 4 columns")
        self._model = model
            
    def __str__(self):
        model = "%s: %s" % ("model".rjust(13), self._print_attr("model"))
        misfit = "%s: %s" % ("misfit".rjust(13), self._print_attr("misfit"))
        n_iter = "%s: %s" % ("n_iter".rjust(13), self._print_attr("n_iter"))
        n_eval = "%s: %s" % ("n_eval".rjust(13), self._print_attr("n_eval"))
        return "\n".join((model, misfit, n_iter, n_eval)) + "\n"
        
    def _print_attr(self, attr):
        if attr not in [ "model", "misfit", "n_iter", "n_eval" ]:
            raise ValueError("attr should be 'model', 'misfit', 'n_iter' or 'n_eval'")
        else:
            if self._model is not None and attr == "model":
                n_lay = len(self._model) // 3
                model = self._model.reshape((n_lay, 3), order = "F")
                param = "\n\t\tVP (m/s)\tVS (m/s)\tThickness (m)\n"
                for i in range(n_lay):
                    param += "\t\t%.2f\t\t%.2f\t\t%.2f\n" % (model[i,0]*model[i,2], model[i,0], model[i,1])
                return param[:-2]
            elif hasattr(self, "_misfit") and attr == "misfit":
                return "%.2f" % self._misfit
            elif hasattr(self, "_n_iter") and attr == "n_iter":
                return "%s" % self._n_iter
            elif hasattr(self, "_n_eval") and attr == "n_eval":
                return "%s" % self._n_eval
            else:
                return None


    def invert3D_fixed_interfaces(self, dcurves_global, dict_disp_curves_tomo,
                                  interfaces, vp_model,
                                  vp_over_vs, ny = 100, dtype = "float32", n_threads = 1,
                                  evo_kws = dict(popsize = 10, max_iter = 100, constrain = True),
                                  opt_kws = dict(solver = "cpso")):
        """Invert the different modes of the dispersion curves measured at a
        set of positions by tomography for a layered 3D velocity model.

        Layer interfaces are supposed known,
        as well as the P wave velocity as function of X,Y in each layer
        (from logs / active seismics)

        Layers' S-wave velocities are determined by the Vp/Vs
        ratio optimized in a user-specified range.

        Layers' densities are determined using the Nafe-Drake's equation as
        they only affect the amplitudes of the dispersion, not the location of
        the zero-crossing.

        Parameters
        ----------
        dcurves_global : list of DispersionCurve
            Dispersion curves to invert as average over all model cells
            (typically a phase velocity from array processing on the full network).
        dict_disp_curves_tomo : dictionnary of gathered tomography results
            as created by make_synthetics.tomo_plugin
            Dispersion curves to invert in each model cell.
            (typically group velocities measured by FTAN + tomography workflow such as passivetomochain).
        vp_over_vs : ndarray (vp_over_vs_min, vp_over_vs_max)
            VP/VS boundaries.
        interfaces : todo define interfaces in docstring
        vp_model : todo define interface in docstring
        ny : int, default 100
            Number of samples on the Y axis.
        dtype : {'float32', 'float64'}, default 'float32'
            Models data type.
        n_threads : int, default 1
            Number of threads to pass to OpenMP for forward modelling.
        evo_kws : dict
            Keywords to pass to evolutionary algorithm initialization.
        opt_kws : dict
            Keywords to pass to evolutionary algorithm optimizer.
            """

        # Check inputs
        if not isinstance(dcurves_global, (list, tuple)) or not np.all(
                [isinstance(dcurve, DispersionCurve) for dcurve in dcurves_global]):
            raise ValueError("dcurves_global must be a list of DispersionCurve objects")
        else:
            self._dcurves = dcurves_global
        if not isinstance(vp_over_vs, np.ndarray) or vp_over_vs.ndim != 2:
            raise ValueError("vp_over_vs must be a 2-D ndarray")
        else:
            self._n_layers = vp_over_vs.shape[0]
        if np.any(vp_over_vs[:, 1] < vp_over_vs[:, 0]):
            raise ValueError("elements in vp_over_vs_max must be greater than vp_over_vs_min")
        if not isinstance(ny, int) or ny < 1:
            raise ValueError("ny must be a positive integer")
        if not isinstance(n_threads, int) or n_threads < 1:
            raise ValueError("n_threads must be a positive integer")
        if not isinstance(opt_kws, dict):
            raise ValueError("opt_kws must be a dictionary")
        if not isinstance(evo_kws, dict):
            raise ValueError("evo_kws must be a dictionary")
        if "constrain" not in evo_kws:
            evo_kws.update(constrain=True)
        else:
            evo_kws["constrain"] = True
        if "eps2" not in evo_kws:
            evo_kws.update(eps2=-1e30)
        else:
            evo_kws["eps2"] = -1e30
        if "snap" not in evo_kws:
            evo_kws.update(snap=True)
        else:
            evo_kws["snap"] = True

        # :todo add input checks for dispersion_tomo_dict, interfaces, vp_model
        # :todo select only valid cells and populate with dispersion curves
        args = ( ny, n_threads )
        lower = vp_over_vs[:,0]
        upper = vp_over_vs[:,1]
        ea = Evolutionary(self._costfunc_3d, lower, upper, args = args, **evo_kws)
        xopt, gfit = ea.optimize(**opt_kws)
        self._misfit = gfit
        self._model = np.array(xopt, dtype = dtype)
        self._misfits = np.array(ea.energy, dtype = dtype)
        self._models = np.array(ea.models, dtype = dtype)
        self._n_iter = ea.n_iter
        self._n_eval = ea.n_eval
        return self

    def invert(self, dcurves, beta, thickness, ny = 100, dtype = "float32", n_threads = 1,
               evo_kws = dict(popsize = 10, max_iter = 100, constrain = True),
               opt_kws = dict(solver = "cpso")):
        """
        Invert the different modes of the dispersion curve for a layered
        velocity model. Layers' P-wave velocities are determined by the Vp/Vs
        ratio ranging in [ 1.5, 2.2 ]. High uncertainties should be expected
        for P-wave velocities as surface waves are not much sensitive to Vp.
        Layers' densities are determined using the Nafe-Drake's equation as
        they only affect the amplitudes of the dispersion, not the location of
        the zero-crossing.
        
        Parameters
        ----------
        dcurves : list of DispersionCurve
            Dispersion curves to invert.
        beta : ndarray (beta_min, beta_max)
            S-wave velocity boundaries in m/s.
        thickness : ndarray (d_min, d_max)
            Layer thickness boundaries in m.
        ny : int, default 100
            Number of samples on the Y axis.
        dtype : {'float32', 'float64'}, default 'float32'
            Models data type.
        n_threads : int, default 1
            Number of threads to pass to OpenMP for forward modelling.
        evo_kws : dict
            Keywords to pass to evolutionary algorithm initialization.
        opt_kws : dict
            Keywords to pass to evolutionary algorithm optimizer.
        """
        # Check inputs
        if not isinstance(dcurves, (list, tuple)) or not np.all([ isinstance(dcurve, DispersionCurve) for dcurve in dcurves ]):
            raise ValueError("dcurves must be a list of DispersionCurve objects")
        else:
            self._dcurves = dcurves
        if not isinstance(beta, np.ndarray) or beta.ndim != 2:
            raise ValueError("beta must be a 2-D ndarray")
        else:
            self._n_layers = beta.shape[0]
        if np.any(beta[:,1] < beta[:,0]):
            raise ValueError("elements in beta_max must be greater than beta_min")
        if not isinstance(thickness, np.ndarray) or thickness.ndim != 2:
            raise ValueError("thickness must be a 2-D ndarray")
        if thickness.shape[0] != self._n_layers:
            raise ValueError("inconsistent number of layers in thickness, got %d instead of %d" \
                             % (thickness.shape[0], self._n_layers))
        if np.any(thickness[:,1] < thickness[:,0]):
            raise ValueError("elements in d_max must be greater than d_min")
        if not isinstance(ny, int) or ny < 1:
            raise ValueError("ny must be a positive integer")
        if not isinstance(n_threads, int) or n_threads < 1:
            raise ValueError("n_threads must be a positive integer")
        if not isinstance(opt_kws, dict):
            raise ValueError("opt_kws must be a dictionary")
        if not isinstance(evo_kws, dict):
            raise ValueError("evo_kws must be a dictionary")
        if "constrain" not in evo_kws:
            evo_kws.update(constrain = True)
        else:
            evo_kws["constrain"] = True
        if "eps2" not in evo_kws:
            evo_kws.update(eps2 = -1e30)
        else:
            evo_kws["eps2"] = -1e30
        if "snap" not in evo_kws:
            evo_kws.update(snap = True)
        else:
            evo_kws["snap"] = True
        
        args = ( ny, n_threads )
        lower = np.concatenate((beta[:,0], thickness[:,0], np.full(self._n_layers, 1.51)))
        upper = np.concatenate((beta[:,1], thickness[:,1], np.full(self._n_layers, 2.19)))
        ea = Evolutionary(self._costfunc, lower, upper, args = args, **evo_kws)
        xopt, gfit = ea.optimize(**opt_kws)
        self._misfit = gfit
        self._model = np.array(xopt, dtype = dtype)
        self._misfits = np.array(ea.energy, dtype = dtype)
        self._models = np.array(ea.models, dtype = dtype)
        self._n_iter = ea.n_iter
        self._n_eval = ea.n_eval
        return self

    def _costfunc_3d(self, x, *args):
        ny, n_threads, weight_global_curves = args

        # add individual misfits per cell
        # vel = params2lay(x)
        misfit = 0.
        count = 0

        modes_all = dict() # keep track of all available modes in the data
        for i_cell in self._n_cells:
            # :todo populate layered model instance with valid cells when calling lm.invert3d
            # :todo populate layered model instance with interfaces when calling lm.invert3d
            # :todo populate layered model instance with dispersion_curve objects per cell when calling lm.invert3d
            dcurves_cell = 'get_tomo_d_curve_from_cell'
            # :todo create or import get_tomo_d_curve_from_cell function
            x_standard = process_fixed_interface_inversion_params(x, self._interfaces, self._vp_model)
            # :todo create process_fixed_interface_inversion_params
            vel = params2lay(x_standard)
            for i, dcurve in enumerate(dcurves_cell):
                th = ThomsonHaskell(vel, dcurve.wtype)
                th.propagate(dcurve.faxis, ny = ny, domain = "fc", n_threads = n_threads)
                if np.any([ np.isnan(sec) for sec in th._panel.ravel() ]):
                    return np.Inf
                else:
                    dc_calc = th.pick([ dcurve.mode ])
                    if (dc_calc[0].npts > 1):
                        if dc_calc[0].faxis[0] <= dcurve.faxis[0]:
                            for dc in dc_calc:
                                dc.dtype = dcurve.dtype
                            nan_count = np.sum(np.isnan(dc_calc[0].dtype_velocity))
                            if nan_count > 0:
                                misfit += np.Inf
                                break
                            else:
                                # if dc.flag_stop:
                                #     print('Debug')
                                dc_obs = np.interp(dc_calc[0].faxis, dcurve.faxis, dcurve.dtype_velocity)
                                n = dc_calc[0].npts
                                dc_unc = dcurve.uncertainties[:n]
                                misfit += np.sum(np.square(dc_obs - dc_calc[0].dtype_velocity)/np.square(dc_unc))
                                count += dcurve.npts
                                if (dcurve.mode not in modes_all.keys()) | \
                                        (dcurve.dtype not in modes_all[dcurve.mode].keys()):
                                    modes_all[dcurve.mode][dcurve.dtype]['faxis'] = dc_calc[0].faxis
                                    modes_all[dcurve.mode][dcurve.dtype]['dcurve_sum'] = dc_calc[0].dtype_velocity
                                    modes_all[dcurve.mode][dcurve.dtype]['ncurves'] = np.ones_like(dc_calc[0].faxis)
                                else:
                                    faxis_old = modes_all[dcurve.mode][dcurve.dtype]['faxis']
                                    dcurve_mean_old = modes_all[dcurve.mode][dcurve.dtype]['dcurve_sum']
                                    ncurves_old = modes_all[dcurve.mode][dcurve.dtype]['ncurves']
                                    faxis_new = dc_calc[0].faxis
                                    dcurve_new = dc_calc[0].dtype_velocity
                                    faxis = np.union1d(faxis_old, faxis_new)
                                    dcurve_mean_old_new_shape = np.zeros_like(faxis)
                                    dcurve_mean_new_new_shape = np.zeros_like(faxis)
                                    ncurves_old_new_shape = np.zeros_like(faxis)
                                    ncurves_new_new_shape = np.zeros_like(faxis)
                                    new_in = np.isin(faxis_new, faxis)
                                    in_new = np.isin(faxis, faxis_new)
                                    old_in = np.isin(faxis_old, faxis)
                                    in_old = np.isin(faxis, faxis_old)
                                    dcurve_mean_old_new_shape[in_old] = dcurve_mean_old[old_in]
                                    ncurves_old_new_shape[in_old] = ncurves_old[old_in]
                                    dcurve_mean_new_new_shape[in_new] = dcurve_new[new_in]
                                    ncurves_new_new_shape[in_new] = 1
                                    modes_all[dcurve.mode][dcurve.dtype]['dcurve_mean'] = dcurve_mean_new_new_shape + \
                                                                            dcurve_mean_old_new_shape
                                    modes_all[dcurve.mode][dcurve.dtype]['ncurves'] = ncurves_old_new_shape + \
                                                                            ncurves_new_new_shape
                        else:
                            misfit += np.Inf
                            break
                    else:
                        misfit += np.Inf
                        break

            # add global misfit
            misfit_global = 0.
            count_global = 0
            for i, dcurve in enumerate(self._dcurves):
                dcurve_calc_mean = modes_all[dcurve.mode][dcurve.dtype]['dcurve_mean'] / \
                                   modes_all[dcurve.mode][dcurve.dtype]['ncurves']
                faxis_calc = modes_all[dcurve.mode][dcurve.dtype]['faxis']

                if len(faxis_calc) > 1:
                    if faxis_calc[0] <= dcurve.faxis[0]:
                        nan_count = np.sum(np.isnan(dcurve_calc_mean))
                        if nan_count > 0:
                            misfit_global += np.Inf
                            break
                        else:
                            dc_obs = np.interp(faxis_calc, dcurve.faxis, dcurve.dtype_velocity)
                            n = len(faxis_calc)
                            dc_unc = dcurve.uncertainties[:n]
                            misfit += np.sum(np.square(dc_obs - dcurve_calc_mean)/np.square(dc_unc))
                            count_global += dcurve.npts
                    else:
                        misfit_global += np.Inf
                        break
                else:
                    misfit_global += np.Inf
                    break

            misfit_global = np.sqrt(misfit_global/count_global)
            misfit = np.sqrt(misfit / count)
            misfit_cumulated = weight_global_curves * misfit_global + (1 - weight_global_curves) * misfit

            if count != 0:
                return misfit_cumulated
            else:
                return np.Inf

    def _costfunc(self, x, *args):
        ny, n_threads = args
        vel = params2lay(x)
        misfit = 0.
        count = 0
        for i, dcurve in enumerate(self._dcurves):
            th = ThomsonHaskell(vel, dcurve.wtype)
            th.propagate(dcurve.faxis, ny = ny, domain = "fc", n_threads = n_threads)
            if np.any([ np.isnan(sec) for sec in th._panel.ravel() ]):
                return np.Inf
            else:
                dc_calc = th.pick([ dcurve.mode ])
                if (dc_calc[0].npts > 1):
                    if dc_calc[0].faxis[0] <= dcurve.faxis[0]:
                        for dc in dc_calc:
                            dc.dtype = dcurve.dtype
                        nan_count = np.sum(np.isnan(dc_calc[0].dtype_velocity))
                        if nan_count > 0:
                            misfit += np.Inf
                            break
                        else:
                            # if dc.flag_stop:
                            #     print('Debug')
                            dc_obs = np.interp(dc_calc[0].faxis, dcurve.faxis, dcurve.dtype_velocity)
                            n = dc_calc[0].npts
                            dc_unc = dcurve.uncertainties[:n]
                            misfit += np.sum(np.square(dc_obs - dc_calc[0].dtype_velocity)/np.square(dc_unc))
                            count += dcurve.npts
                    else:
                        misfit += np.Inf
                        break
                else:
                    misfit += np.Inf
                    break
        if count != 0:
            return np.sqrt(misfit / count)
        else:
            return np.Inf
    
    def params2lay(self):
        """
        Convert parameters to a layered velocity model usable by ThomsonHaskell
        object.
        
        Returns
        -------
        vel : ndarray
            Layered velocity model. Each row defines the layer parameters in
            the following order: P-wave velocity (m/s), S-wave velocity (m/s),
            density (kg/m3) and thickness (m).
        """
        return params2lay(self._model)
    
    def params2vel(self, vtype = "s", nz = 100, zmax = None):
        """
        Convert parameters to a continuous layered velocity model.
        
        Parameters
        ----------
        vtypes : {'s', 'p'}, default 's'
            Velocity model type.
        nz : int, default 100
            Number of depth discretization points.
        zmax : float or None, default None
            Maximum depth.
        
        Returns
        -------
        vel : ndarray
            Layered velocity model. Each row defines the layer parameters in
            the following order: P-wave velocity (m/s), S-wave velocity (m/s),
            density (kg/m3) and thickness (m).
        """
        return params2vel(self._model, vtype, nz, zmax)
    
    def panel(self, wtype = "rayleigh", nf = 200,
              th_kws = dict(ny = 200, domain = "fc", n_threads = 1)):
        """
        Compute the Thomson-Haskell panel.
        
        Parameters
        ----------
        wtype : {'rayleigh', 'love'}, default 'rayleigh'
            Surface wave type.
        nf : int, default 200
            Number of frequency samples.
        th_kws : dict
            Keyworded arguments passed to ThomsonHaskell propagate method.
        
        Returns
        -------
        th : ThomsonHaskell
            Dispersion curve panel.
        """
        faxis = [ dcurve.faxis for dcurve in self._dcurves ]
        faxis_full = np.unique(np.concatenate([ f for f in faxis ]))
        faxis_new = np.linspace(faxis_full.min(), faxis_full.max(), nf)
        vel = self.params2lay()
        th = ThomsonHaskell(vel, wtype)
        th.propagate(faxis_new, **th_kws)
        return th
    
    def pick(self, modes = [ 0 ], wtype = "rayleigh", nf = 200,
             th_kws = dict(ny = 200, domain = "fc", n_threads = 1)):
        """
        Parameters
        ----------
        modes : list of int, default [ 0 ]
            Modes number to pick (0 if fundamental).
        wtype : {'rayleigh', 'love'}, default 'rayleigh'
            Surface wave type.
        nf : int, default 200
            Number of frequency samples.
        th_kws : dict
            Keyworded arguments passed to ThomsonHaskell propagate method.
        
        Returns
        -------
        picks : list of DispersionCurve
            Picked dispersion curves.
        """
        th = self.panel(wtype, nf, th_kws)
        return th.pick(modes)
    
    def save(self, filename):
        """
        Pickle the dispersion curve to a file.
        
        Parameters
        ----------
        filename: str
            Pickle filename.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    @property
    def model(self):
        if hasattr(self, "_model"):
            return self._model
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def misfit(self):
        if hasattr(self, "_misfit"):
            return self._misfit
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def models(self):
        if hasattr(self, "_models"):
            return self._models
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def misfits(self):
        if hasattr(self, "_misfits"):
            return self._misfits
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def energy(self):
        energy = np.min(self.misfits, axis = 0)
        return np.array([ np.min(energy[:i+1]) for i in range(len(energy)) ])
            
    @property
    def n_iter(self):
        if hasattr(self, "_n_iter"):
            return self._n_iter
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def n_eval(self):
        if hasattr(self, "_n_eval"):
            return self._n_eval
        else:
            raise AttributeError("no inversion performed yet")
            

def _betanu2alpha(beta, nu):
    return beta * np.sqrt( np.abs( ( 1.-nu ) / ( 0.5 - nu ) ) ) 

    
def _nafe_drake(alpha):
    coeff = np.array([ 1.6612, -0.4712, 0.0671, -0.0043, 0.000106 ])
    alpha_pow = np.array([ alpha*1e-3, (alpha* 1e-3)**2, (alpha*1e-3)**3,
                          (alpha*1e-3)**4, (alpha*1e-3)**5 ])
    return np.dot(coeff, alpha_pow) * 1e3
            
            
def params2lay(x):
    """
    Convert parameters to a layered velocity model usable by ThomsonHaskell
    object.
    
    Parameters
    ----------
    x : ndarray
        Array of parameters.
    
    Returns
    -------
    vel : ndarray
        Layered velocity model. Each row defines the layer parameters in
        the following order: P-wave velocity (m/s), S-wave velocity (m/s),
        density (kg/m3) and thickness (m).
    """
    n_layers = len(x) // 3
    beta = x[:n_layers]
    alpha = beta * x[2*n_layers:]
    rho = _nafe_drake(alpha)
    d = x[n_layers:2*n_layers]
    vel = np.concatenate((alpha[:,None], beta[:,None], rho[:,None], d[:,None]), axis = 1)
    return vel


def params2vel(x, vtype = "s", nz = 100, zmax = None):
    """
    Convert parameters to a continuous layered velocity model.
    
    Parameters
    ----------
    x : ndarray
        Array of parameters.
    vtypes : {'s', 'p'}, default 's'
        Velocity model type.
    nz : int, default 100
        Number of depth discretization points.
    zmax : float or None, default None
        Maximum depth.
    
    Returns
    -------
    vel : ndarray
        Layered velocity model. Each row defines the layer parameters in
        the following order: P-wave velocity (m/s), S-wave velocity (m/s),
        density (kg/m3) and thickness (m).
    """
    lay = params2lay(x)
    zint = np.cumsum(lay[:,-1])
    if zmax is None:
        thick_min = lay[:,-1].min()
        zmax = zint[-2] +  thick_min
    zint[-1] = zmax
    dz = zmax / nz
    az = dz * np.arange(nz)
    if vtype == "s":
        layz = np.stack((lay[:,1], zint)).transpose()
    elif vtype == "p":
        layz = np.stack((lay[:,0], zint)).transpose()
    else:
        raise ValueError("unknown velocity type '%s'" % vtype)
    vel = l2vf.lay2vel1(layz, dz, nz)
    return vel, az

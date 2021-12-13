# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .dispersion_curve import DispersionCurve
from ._dispcurve import dispcurve as dc

__all__ = [ "ThomsonHaskell" ]


class ThomsonHaskell:

    _WTYPE = [ "rayleigh", "love" ]

    """
    Thomson-Haskell propagator.
    
    This class computes the analytical dispersion curve modes in a stratified
    medium.
    
    Parameters
    ----------
    velocity_model : ndarray
        Velocity model. Each row defines the layer parameters in the following
        order: P-wave velocity (m/s), S-wave velocity (m/s), density (kg/m3)
        and thickness (m).
    wtype : {'rayleigh', 'love'}, default 'rayleigh'
        Surface wave type.
    """
    def __init__(self, velocity_model, wtype = "rayleigh"):
        if not isinstance(velocity_model, np.ndarray) or velocity_model.ndim != 2:
            raise ValueError("velocity_model must be a 2-D ndarray")
        if velocity_model.shape[1] != 4:
            raise ValueError("velocity_model must have 4 columns")
        else:
            self._velocity_model = velocity_model
        ratio = velocity_model[:,0] / velocity_model[:,1]
        if np.any(ratio < 1.4) or np.any(ratio > 2.6):
            raise ValueError("non physical Vp/Vs ratio")
        if wtype not in self._WTYPE:
            raise ValueError("wtype must be in %s, got '%s'" % (self._WTYPE, wtype))
        else:
            self._wtype = wtype

    def propagate(self, f, ny = 100, domain = "fc", eps = 0.1, n_threads = 1):
        """
        Compute the analytical dispersion curve modes.
        
        Parameters
        ----------
        f : ndarray
            Frequency axis (in Hz).
        ny : int, default 100
            Number of samples on the Y axis.
        domain : {'fc', 'fk'}, default 'fc'
            Domain in which the dispersion curve is computed:
            - 'fc', phase velocity (m/s).
            - 'fk', wavenumber (rad/m).
        eps : float, default 0.1
            Percentage of expand for Y axis.
        n_threads : int, default 1
            Number of threads to pass to OpenMP.
            
        Returns
        -------
        panel : ndarray
            Dispersion curve panel.
        """
        # Check inputs
        if not isinstance(f, np.ndarray) or f.ndim != 1:
            raise ValueError("f must be a 1-D ndarray")
        if not isinstance(ny, int) or ny < 1:
            raise ValueError("ny must be a positive integer")
        if not isinstance(domain, str) or domain not in [ "fc", "fk" ]:
            raise ValueError("domain should either be 'fc' or 'fk'")

        # Import parameters
        alpha = self._velocity_model[:,0]
        beta = self._velocity_model[:,1]
        rho = self._velocity_model[:,2]
        d = self._velocity_model[:,3]

        # Thomson-Haskell method
        vr = self._rayleigh_velocity()
        vmin = max(0.1, np.floor(vr.min()))
        vmax = beta.max()
        dv = eps * (vmax - vmin)
        vmin = max(0.1, vmin - dv)
        if domain == "fc":
            y = np.linspace(vmin, vmax, ny)
            panel = dc.fcpanel(f, y, alpha, beta, rho, d,
                               wtype = self._wtype, n_threads = n_threads)
        elif domain == "fk":
            y = np.linspace(2.*np.pi*f[0]/vmax, 2.*np.pi*f[-1]/vmin, ny)
            panel = dc.fkpanel(f, y, alpha, beta, rho, d,
                               wtype = self._wtype, n_threads = n_threads)

        self._faxis = f
        self._yaxis = y
        self._panel = np.real(panel)
        self._domain = domain
        return panel

    def _poisson_ratio(self):
        alpha = self._velocity_model[:,0]
        beta = self._velocity_model[:,1]
        ksi = alpha**2 / beta**2
        return ( 1. - 0.5*ksi ) / ( 1. - ksi )

    def _rayleigh_velocity(self):
        beta = self._velocity_model[:,1]
        nu = self._poisson_ratio()
        return beta * ( 0.87 + 1.12 * nu ) / ( 1. + nu )

    def pick(self, modes = [ 0 ]):
        """
        Pick dispersion curve for different propagation mode.
        
        Parameters
        ----------
        modes : list of int, default [ 0 ]
            Modes number to pick (0 if fundamental).
            
        Returns
        -------
        dcurves : list of DispersionCurve
            Picked dispersion curves.
        """
        if not hasattr(self, "_panel"):
            raise ValueError("no propagation performed yet")
        if self._domain == "fk":
            raise ValueError("cannot perform picking in FK domain")
        if not isinstance(modes, list) or np.min(modes) < 0 \
            or not np.all([ isinstance(m, int) for m in modes ]):
            raise ValueError("modes must be a list of positive integers")

        dcurve = [ [] for m in modes ]
        faxis = [ [] for m in modes ]

        n_mode = 0                              # mode counter
        count_jump = 0                          # counter of new modes appearing in the panel since
                                                # the last detection of a lacking mode
        i_max_search = self._panel.shape[0]     # index to cut the panel for mode detection only in useful range
        for i, f in enumerate(self._faxis):
            tmp = self._panel[:i_max_search, i] / \
                  np.abs(np.nanmax(self._panel[:i_max_search, i]))      # extract panel column
            idx = np.where((np.sign(tmp[:-1]) *
                            np.sign(tmp[1:])) < 0.)[0]                  # detect sign changes = modes
            tmp_non_nan = tmp[~np.isnan(tmp)]
            sgn = np.sign(tmp_non_nan[-1])                              # store sign on the top of the panel
                                                                        # (useful for detecting when a new mode enters)
            if i == 0:
                sgn0 = sgn                                              # initialize sign
            n_mode_new = len(idx)                                       # count detected modes
            condition1 = n_mode_new > (n_mode + count_jump)
                        # new number of modes is greater than previous
            condition2 = (sgn == sgn0) & (n_mode_new == n_mode + count_jump)
                        # new number of modes is equal to the previous
                        # and the sign at the top of the panel hasn't changed
            regular_situation_condition = condition1 | condition2
            if regular_situation_condition:
                for j, m in enumerate(modes):
                    if len(idx) >= m+1:
                        xr = [ tmp[idx[m]], tmp[idx[m]+1] ]
                        vr = [ self._yaxis[idx[m]], self._yaxis[idx[m]+1] ]
                        x = ( vr[0] * xr[1] - vr[1] * xr[0] ) / ( xr[1] - xr[0] )
                        dcurve[j].append(x)
                        faxis[j].append(f)

                if (n_mode_new > len(modes) + 1):
                    # cut the panel above the mode mode_max_requested + 1
                    i_max_search = min(idx[len(modes)] + 2, self._panel.shape[0] - 1)
                    # update everything
                    tmp = self._panel[:i_max_search, i] / np.abs(np.nanmax(self._panel[:i_max_search, i]))
                    idx = np.where((np.sign(tmp[:-1]) * np.sign(tmp[1:])) < 0.)[0]
                    tmp_non_nan = tmp[~np.isnan(tmp)]
                    n_mode_new = len(idx)
                    sgn = np.sign(tmp_non_nan[-1])

                count_jump = 0
                sgn0 = sgn
                n_mode = n_mode_new
            else:
                if sgn != sgn0:
                    count_jump += 1
                #print("debug")
        for j, m in enumerate(modes):
            if len(faxis[j]) > 0:
                faxis_full = self._faxis[(self._faxis>=min(faxis[j])) & (self._faxis<=max(faxis[j]))]
                dcurve_full = np.interp(faxis_full,faxis[j],dcurve[j])
                faxis[j] = faxis_full
                dcurve[j] = dcurve_full



        dcurves = []

        for i, (d, f) in enumerate(zip(dcurve, faxis)):
            d = DispersionCurve(d, f, int(modes[i]), None, self._wtype, dtype="phase")
            # if len(d.dtype_velocity) == 0 :
            #     print('no dc picked')
            dcurves.append(d)


        #        dcurve = dc.pick(self._panel, self._faxis, self._yaxis, modes, n_threads = n_threads)
#        dcurves = []
#        for i, df in enumerate(dcurve):
#            d, f = df
#            idx = np.where(d > 0.)[0]
#            dcurves.append(DispersionCurve(d[idx], f[idx], int(modes[i]), self._wtype))
        return dcurves

    def save_picks(self, filename, modes = [ 0 ]):
        """
        Export picked dispersion curves to ASCII file.
        
        Parameters
        ----------
        filename : str
            Output file name.
        modes : list of int
            Modal curves to export.
        """
        dcurves = self.pick(modes)
        fid = open(filename, "w")
        for dcurve in dcurves:
            if dcurve.npts > 0:
                if dcurve.mode == 0:
                    header = "# Fundamental mode\n"
                else:
                    header = "# Mode %d\n" % dcurve.mode
                fid.write(header)

                d = dcurve.phase_velocity
                f = dcurve.faxis
                npts = len(d)
                for i in range(npts):
                    fid.write(str(f[i]) + " " + str(d[i]) + "\n")
                fid.write("\n")
        fid.close()

    def plot(self, n_levels = 200, axes = None, figsize = (8, 8), cmap = None,
             cont_kws = {}):
        """
        Plot the dispersion curve panel.
        
        Parameters
        ----------
        n_levels: int, default 200
            Number of levels for contour.
        axes : matplotlib axes or None, default None
            Axes used for plot.
        figsize : tuple, default (8, 8)
            Figure width and height if axes is None.
        cmap : str, default "viridis"
            Colormap.
        cont_kws : dict
            Keyworded arguments passed to contour plot.
            
        Returns
        -------
        cax : matplotlib contour
            Contour plot.
        """
        if not hasattr(self, "_panel"):
            raise ValueError("no propagation performed yet")
        if axes is not None and not isinstance(axes, Axes):
            raise ValueError("axes must be Axes")
        if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple with 2 elements")
        if not isinstance(cont_kws, dict):
            raise ValueError("cont_kws must be a dictionary")

        if cmap is None:
            cmap = self._set_cmap()
        if axes is None:
            fig = plt.figure(figsize = figsize, facecolor = "white")
            fig.patch.set_alpha(0.)
            ax1 = fig.add_subplot(1, 1, 1)
        else:
            ax1 = axes
        cax = ax1.contourf(self._faxis, self._yaxis, np.log(np.abs(self._panel)),
                     n_levels, cmap = cmap, **cont_kws)
        ax1.set_xlabel("Frequency (Hz)", fontsize = 12)
        if self._domain == "fc":
            ax1.set_ylabel("Phase velocity (m/s)", fontsize = 12)
        elif self._domain == "fk":
            ax1.set_ylabel("Wavenumber (rad/m)", fontsize = 12)
        ax1.grid(True, linestyle = ":")
        return cax

    def _set_cmap(self):
        import matplotlib.cm as cm
        if hasattr(cm, "viridis"):
            return "viridis"
        else:
            return "jet"

    @property
    def velocity_model(self):
        """
        ndarray
        Velocity model. Each row defines the layer parameters in the following
        order: P-wave velocity (m/s), S-wave velocity (m/s), density (kg/m3)
        and thickness (m).
        """
        return self._velocity_model

    @velocity_model.setter
    def velocity_model(self, value):
        self._velocity_model = value

    @property
    def wtype(self):
        """
        str
        Surface wave type ('rayleigh' or 'love').
        """
        return self._wtype

    @wtype.setter
    def wtype(self, value):
        self._wtype = value

    @property
    def faxis(self):
        """
        ndarray
        Frequency axis (in Hz).
        """
        return self._faxis

    @faxis.setter
    def faxis(self, value):
        self._faxis = value

    @property
    def uncertainties(self):
        """
        list or ndarray
        Uncertainty on the observed phase velocity.
        """
        return self._uncertainties

    @uncertainties.setter
    def uncertainties(self, value):
        self._uncertainties = value

    @property
    def yaxis(self):
        """
        ndarray
        Y axis:
        - Phase velocity (m/s) if domain = 'fc'.
        - Wavenumber (rad/m) if domain = 'fk'.
        """
        return self._yaxis

    @yaxis.setter
    def yaxis(self, value):
        self._yaxis = value

    @property
    def panel(self):
        """
        ndarray
        Dispersion curve panel.
        """
        return self._panel

    @panel.setter
    def panel(self, value):
        self._panel = value

    @property
    def domain(self):
        """
        Domain in which the dispersion curve is computed.
        """
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value
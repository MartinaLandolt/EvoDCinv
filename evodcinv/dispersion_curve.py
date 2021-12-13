# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import h5py
    
__all__ = [ "DispersionCurve" ]

    
class DispersionCurve:
    
    _WTYPE = [ "rayleigh", "love" ]
    _DTYPE = [ "group", "phase" ]
    
    """
    Dispersion curve.
    
    Parameters
    ----------
    phase_velocity : list or ndarray
        Observed phase velocities (in m/s).
    faxis : list or ndarray
        Frequency axis (in Hz).
    mode : int
        Mode number (0 if fundamental).
    wtype : {'rayleigh', 'love'}, default 'rayleigh'
        Surface wave type.
    dtype : {'phase', 'group'}, default 'phase'
        Measured velocity type
    """
    def __init__(self, velocity, faxis, mode, uncertainties = None, wtype = "rayleigh", dtype="phase"):
        if not isinstance(velocity, (list, np.ndarray)) or np.asanyarray(velocity).ndim != 1:
            raise ValueError("velocity must be a list of 1-D ndarray")
        if not all([ np.min(c) > 0. for c in velocity ]):
            raise ValueError("velocities must be positive")
        else:
            if dtype == "phase":
                self._phase_velocity = velocity
                self._group_velocity = None
            elif dtype == "group":
                self._phase_velocity = None
                self._group_velocity = velocity
            else:
                raise ValueError("dtype must be in %s, got '%s'" % (self._DTYPE, dtype))
            self._npts = len(velocity)
            self._dtype = dtype

        if not isinstance(faxis, (list, np.ndarray)) or np.asanyarray(faxis).ndim != 1 \
            or len(faxis) != self._npts:
            raise ValueError("phase_velocity must be a list of 1-D ndarray of length %d" % self._npts)
        if not np.all([ np.min(f) >= 0. for f in faxis ]):
            raise ValueError("frequencies must be positive")
        else:
            self._faxis = faxis
        if uncertainties is not None:
            if not isinstance(uncertainties, (list, np.ndarray)) or np.asanyarray(uncertainties).ndim != 1 \
                or len(uncertainties) != self._npts:
                raise ValueError("uncertainties must be a list of 1-D ndarray of length %d" % self._npts)
            if not np.all([ np.min(u) >= 0. for u in uncertainties ]):
                raise ValueError("uncertainties must be positive")
            else:
                self._uncertainties = uncertainties

        if not isinstance(mode, int) or mode < 0:
            raise ValueError("mode must be a positive integer")
        else:
            self._mode = mode
        if wtype not in self._WTYPE:
            raise ValueError("wtype must be in %s, got '%s'" % (self._WTYPE, wtype))
        else:
            self._wtype = wtype

    @classmethod
    def from_h5(cls, h5_path, coordinates, mode, wtype="rayleigh", dtype="group"):
        """
        Load dispersion curve using h5 file collecting all dispersion curves
        for a given analysis

        :param coordinates: Array index of the dispersion curve to load.
        :type coordinates: list of 2 int
        """

        with h5py.File(h5_path, 'r') as fin:
            freq = fin["Frequency"][:]
            vel = fin["DispersionCurve"][coordinates[0], coordinates[1], :]
            uncertainties = fin["Uncertainties"][coordinates[0], coordinates[1], :]

        return cls(vel, freq, mode, uncertainties, wtype, dtype)

    @classmethod
    def from_h5_tmp(cls, h5_path, coordinates, mode, wtype="rayleigh", dtype="group"):
        """to get one dispersion curve from a data.h5 file"""
        with h5py.File(h5_path, 'r') as fin:
            freq = fin["Frequency"][:]
            vel = fin["DispersionCurve"][coordinates, :]
            uncertainties = fin["Uncertainties"][coordinates, :]

        return cls(vel, freq, mode, uncertainties, wtype, dtype)
            
    def save(self, filename = None, fmt = "%.8f"):
        """
        Export dispersion curve to ASCII file.
        
        Parameters
        ----------
        filename : str or None, default None
            Output file name.
        fmt : str, default "%.8f"
            ASCII format.
        """
        if filename is None:
            filename = "%s_mode%d.txt" % (self._wtype, self._mode)
        X = np.stack((self._faxis, self.dtype_velocity), axis = 1)
        np.savetxt(filename, X, fmt)
            
    def plot(self, axes = None, figsize = (8, 8), plt_kws = {}):
        """
        Plot the dispersion curve.
        
        Parameters
        ----------
        axes : matplotlib axes or None, default None
            Axes used for plot.
        figsize : tuple, default (8, 8)
            Figure width and height if axes is None.
        plt_kws : dict
            Keyworded arguments passed to line plot.
            
        Returns
        -------
        lax : matplotlib line plot
            Line plot.
        """
        if axes is not None and not isinstance(axes, Axes):
            raise ValueError("axes must be Axes")
        if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple with 2 elements")
        if not isinstance(plt_kws, dict):
            raise ValueError("plt_kws must be a dictionary")
            
        if axes is None:
            fig = plt.figure(figsize = figsize, facecolor = "white")
            fig.patch.set_alpha(0.)
            ax1 = fig.add_subplot(1, 1, 1)
        else:
            ax1 = axes
        lax = ax1.plot(self._faxis, self.dtype_velocity, **plt_kws)
        return lax

    def to_group_velocity(self):
        """
        Convert phase velocity to group velocity by
        differentiating phase_velocity

        Only works if frequencies are evenly spaced.
        """
        if isinstance(self.phase_velocity, list):
            phase_velocity = np.array(self.phase_velocity, dtype=float)
        elif self.phase_velocity is not None:
            phase_velocity = self.phase_velocity
        else:
            raise ValueError("""trying to convert undefined phase velocity 
                    to group velocity""")

        if isinstance(self.faxis, list):
            faxis = np.array(self.faxis, dtype=float)
        else:
            faxis = self.faxis

        if len(faxis) >= 2 :
            # Interpolate phase_velocity on an faxis with small steps to avoid spikes while deriving
            fmin = faxis[0] ; fmax = faxis[-1]
            df = 0.01
            tmp_faxis = np.arange(fmin, fmax+df, df)
            tmp_phase_velocity = np.interp(tmp_faxis, faxis, phase_velocity)

            # omega = 2*np.pi*faxis
            omega = 2 * np.pi * tmp_faxis
            domega = omega[1] - omega[0]
            if not np.allclose(np.diff(omega), domega, rtol=10**-2):
                raise ValueError("""Frequencies not evenly spaced. 
                       Could not convert from phase velocity to group velocity""")
            dphase_domega = np.gradient(tmp_phase_velocity, domega)
            #group_velocity = phase_velocity  + omega * dphase_domega
            group_velocity_tmp = tmp_phase_velocity / (1 - omega/tmp_phase_velocity * dphase_domega)
            diff_vg = np.diff(group_velocity_tmp)
            diff_vg = np.hstack((diff_vg, diff_vg[-1]))
            flag_stop = np.max(np.abs(diff_vg[faxis > 1])) > 100
            # if flag_stop:
            #     print('debug')
            #Interpolate velocity on faxis from the phase_velocity
            group_velocity = np.interp(faxis, tmp_faxis, group_velocity_tmp)

            self.group_velocity = group_velocity
            self.dtype = 'group'
            self.dtype_velocity = group_velocity
            #REMOOOVE AFTER DEBUG
            self.flag_stop = flag_stop
        else :
            raise ValueError("""dipersion curve to small (empty or only one value) to convert to groupe velocity""")

    @property
    def dtype_velocity(self):
        """Returns group or phase velocity, such as specified by dtype"""
        dtype_vel = '_'+self.dtype+'_velocity'
        return getattr(self, dtype_vel)

    @dtype_velocity.setter
    def dtype_velocity(self, value):
        dtype_vel = '_'+self.dtype+'_velocity'
        setattr(self, dtype_vel, value)

    @property
    def phase_velocity(self):
        """
        list or ndarray
        Observed phase velocities (in m/s).
        """
        return self._phase_velocity

    @property
    def group_velocity(self):
        """
        ndarray
        Observed group velocities (in m/s).
        """
        return self._group_velocity
    
    @phase_velocity.setter
    def phase_velocity(self, value):
        self._phase_velocity = value
        
    @group_velocity.setter
    def group_velocity(self, value):
        self._group_velocity = value

    @property
    def faxis(self):
        """
        list or ndarray
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

    def interpolate_faxis(self, value):
        """Interpolates velocity for a given frequency axis"""
        self.dtype_velocity = np.interp(value, self.faxis, self.dtype_velocity)
        self.faxis = value

    @property
    def mode(self):
        """
        int
        Mode number.
        """
        return self._mode
    
    @mode.setter
    def mode(self, value):
        self._mode = value
    
    @property
    def wtype(self):
        """
        str
        Surface wave type.
        """
        return self._wtype
    
    @wtype.setter
    def wtype(self, value):
        self._wtype = value

    @property
    def dtype(self):
        """
        str
        Type of velocity stored (phase or group)
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if value not in self._DTYPE:
            raise ValueError("dtype must be in %s, got '%s'" % (self._DTYPE, dtype))
        self._dtype = value

        if self.dtype_velocity is None and value=="group":
            self.to_group_velocity()
        elif self.dtype_velocity is None and value=="phase":
            raise ValueError("Impossible to convert from group velocity to phase velocity")
        
    @property
    def npts(self):
        """
        int
        Number of points that define the dispersion curve.
        """
        return self._npts

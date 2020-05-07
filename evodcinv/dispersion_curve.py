# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
    
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
    """
    def __init__(self, velocity, faxis, mode, wtype = "rayleigh", dtype="phase"):
        if not isinstance(velocity, (list, np.ndarray)) or np.asanyarray(phase_velocity).ndim != 1:
            raise ValueError("velocity must be a list of 1-D ndarray")
        if not all([ np.min(c) > 0. for c in velocity ]):
            raise ValueError("velocities must be positive")
        else:
            if dtype == "phase":
                self._phase_velocity = velocity
                self._group_velocity = None
            elif dtype == "group"
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
        if not isinstance(mode, int) or mode < 0:
            raise ValueError("mode must be a positive integer")
        else:
            self._mode = mode
        if wtype not in self._WTYPE:
            raise ValueError("wtype must be in %s, got '%s'" % (self._WTYPE, wtype))
        else:
            self._wtype = wtype
            
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
        lax = ax1.plot(self._faxis, self.dtype_velocity)
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
            raise ValueError("trying to convert undefined phase velocity 
                    to group velocity")
        
        if isinstance(self.faxis, list):
            faxis = np.array(self.faxis, dtype=float)
        else:
            faxis = self.faxis

        omega = 2*np.pi*faxis
        domega = omega[1] - omega[0]
        if not np.allclose(np.diff(omega), domega): 
            raise ValueError("Frequencies not evenly spaced.
                    Could not convert from phase velocity to group velocity")
        dphase_domega = np.gradient(phase_velocity, domega) 
        group_velocity = phase_velocity  + omega * dphase_domega

        self.group_velocity = group_velocity
    
    @property
    def dtype_velocity(self):
        """Returns group or phase velocity, such as specified by dtype"""
        dtype_vel = '_'+self.dtype+'_velocity'
        return getattr(self, dtype_vel)

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
        self._dtype = dtype

        if self.dtype_velocity is None and dtype=="group":
            self.to_group_velocity()
        elif self.dtype_velocity is None and dtype=="phase"
            raise ValueError("Impossible to convert from group velocity to phase velocity")
        
    @property
    def npts(self):
        """
        int
        Number of points that define the dispersion curve.
        """
        return self._npts

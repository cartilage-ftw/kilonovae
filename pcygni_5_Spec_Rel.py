#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2017 Ulrich Noebauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EveNT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
References
----------
 * Jeffery & Branch 1990: "Analysis of Supernova Spectra"
   ADS link:http://adsabs.harvard.edu/abs/1990sjws.conf..149J
 * Thomas et al 2011: "SYNAPPS: Data-Driven Analysis for Supernova
   Spectroscopy"
   ADS link:http://adsabs.harvard.edu/abs/2011PASP..123..237T
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import scipy
import astropy.units as units
import astropy.constants as csts
from numba.experimental import jitclass
import numba 
from multiprocessing import Pool
import multiprocessing

spec = [
    ('rmax', numba.float32), 
    ('rmin', numba.float32), 
    ('Ip', numba.float32), 
    ('t', numba.float32), 
    ('vdet_min', numba.float32), 
    ('vdet_max', numba.float32), 
    ('tauref', numba.float32), 
    ('vref', numba.float32), 
    ('ve', numba.float32), 
]

def proxy(f):
    return lambda x: f(x)
    

@numba.njit
def _calc_p( r, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve):
    """
    Calculate p-coordinate of location (r,z) in ejecta

    Parameters
    ----------
    r : float
        radial coordinate of location of interest
    z : float
        z-coordinate (i.e. along the line-of-sight to the observer) of the
        location of interest

    Returns
    -------
    p : float
        p-coordinate (perpendicular to z) of the location of interest
    """

    assert(np.abs(r) > np.abs(z))

    return np.sqrt(r**2 - z**2)

@numba.njit
def _calc_r( p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve):
    """
    Calculate radius of location (z, p) in ejecta;

    Parameters
    ----------
    p : float
        p-coordinate (perpendicular to line-of-sight to observer)
    z : float
        z-coordinate (along line-of-sight to observer)

    Returns
    -------
    r : float
        radius of location
    """
    return np.sqrt(p**2 + z**2)

@numba.njit
def _calc_W(r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve):
    """
    Calculate geometric dilution factor

    Parameters
    ----------
    r : float
        radius of location

    Returns
    -------
    W : float
        geometric dilution factor
    """

    return (np.float32(1) - np.sqrt(np.float32(1) - (vphot*t / r)**2)) / 2
 
@numba.njit
def _calc_tau(r, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, c):
    """
    Calculate line optical depth at radius r, according to density profile.

    We assume an exponential density and thus optical depth profile as
    presented in Thomas et al. 2011.

    Arguments
    ---------
    r : float
        radius of location

    Returns
    -------
    tau : float
        line optical depth
    """

    v = r / t

    #Correction from "Formation of P Cygni Line Profiles in Relativistically Expanding Atmospheres" by Hutsemékers and Surdej (1990).
    mu = z/r
    beta = v/c
    corr = abs( (1-mu*beta)**2/((1-beta)*(mu*(mu-beta)+(1-mu**2)*(1-beta**2))) )
    #print(corr)
    
    if v >= vdet_min and v <= vdet_max:
        return corr*tauref * np.exp((vphot - v)/ve)
    else:
        return np.float32(1e-20)

@numba.njit
def _S(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, c, delta):
    """
    Calculate source function at location (p, z) in ejecta.

    In case only the pure absorption component of the line profile is
    considered, the source function is of course 0. Otherwise, it follows
    from eq. 33 of Jeffery & Branch 1990.

    Parameters
    ----------
    p : float
        p-coordinate of location
    z : float
        z-coordinate of location
    mode : str
        flag setting the interaction mode: 'both' for full line profile,
        'abs' for pure absorption (default 'both')

    Returns
    -------
    S : float
        source function at location (p, z)
    """

    r = _calc_r(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve)

    if r > vmax*t or r < vphot*t:
        # outside ejecta or inside photosphere
        return 0
    elif z < 0 and p < vphot*t:
        # occulted region
        return 0
    else:
        # emission region
        
        #purposed source-function correction
        #Correction from "Formation of P Cygni Line Profiles in Relativistically Expanding Atmospheres" by Hutsemékers and Surdej (1990).

        #costheta = z/r
        #beta = r/(t*c)
        #delta_b = (1-beta**2)**(1/2) /(1+beta*costheta) 
        #print('cos = ', costheta, ', delta = ', delta_b)
        #Note the sign is implicit in cos(theta) as z is generically negative towards us
        
        W = _calc_W(r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve) * (delta**2)
        return W * _I(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve)

@numba.njit
def _I(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve):
    """
    Determine the initial specific intensity for a ray passing through (p,
    z) towards the observer.

    Used in eq. 71 of Jeffery & Branch 1990. Only if the line of sight
    going through (p, z) and towards the observer intersects the
    photosphere, a non-zero initial specific intensity is found.

    Parameters
    ----------
    p : float
        p-coordinate of location of interest
    z : float
        z-coordinate of location of interest

    Returns
    -------
    I : float
        initial specific intensity
    """

    if p < vphot*t:
        # in the photosphere plane
        return 1 # TODO: modify this continuum to a Planck, or user-decided continuum, instead of constant (flat)
    else:
        # above the photosphere plane
        return 0

@numba.njit
def _tau(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, c):
    """
    Determine the line optical on the line-of-sight towards the observer,
    at location (p, z).

    Used in eq.  of Jeffery & Branch 1990. Only locations in the emission
    region outside of the occulted zone may attenuated the radiation field.
    Thus, only there a non-zero optical depth is returned.

    Parameters
    ----------
    p : float
        p-coordinate of the location of interest
    z : float
        z-coordinate of the location of interest

    Returns
    -------
    tau : float
        optical depth at the location of interest
    """

    r = _calc_r(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve)

    if r > vmax*t or r < vphot*t:
        # outside ejecta or inside photosphere
        return 0
    elif z < 0 and p < vphot*t:
        # occulted region
        return 0
    else:
        # emission region        
        return _calc_tau(r, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, c)

@numba.njit(fastmath=True, nogil=True)
def _Iemit(p, delta, vmax, vphot, t, vdet_min, vdet_max, tauref, ve):#, mode="both"):
    """
    Determine the total specific intensity eventually reaching the observer
    from (p, nu).

    The absorption or emission-only cases may be treated, or both effects
    may be included to calculate the full line profile. Used in eq. 71 of
    Jeffery & Branch 1990.

    Parameters
    ----------
    p : float
        p-coordinate of location of interest
    z : float
        z-coordinate of location of interest
    mode : str
        flag determining the line profile calculation mode: 'abs' for pure
        absorption, 'emit' for pure emission, 'both' for the full line
        profile calculation (default 'both')

    Returns
    -------
    Iemit : float
        total specific intensity emitted towards the observer from
        location (p, z)
    """
    c = 29979245800.0
    #z = (1-1/delta)*c*t
    #z = c*t* (delta**2/(1+delta**2) - delta**2/(1+delta**2) * (1-(1+delta**2)/delta**4*( (p/c/t)**2-(1-delta**2)))**(1/2))
    z = c*t* (delta**2/(1+delta**2) - delta**2/(1+delta**2) * (1+(1+delta**2)/delta**4*( (1-delta**2-(p/c/t)**2)))**(1/2))
    
    #print(z/(c*t), p/(c*t))
    
    tau = _tau(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, c)

    #if mode == "both" or mode == "abs":
    return (_I(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve) * np.exp(-tau) + _S(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, c, delta) *
                (np.float32(1) - np.exp(-tau))) * p
    #else:
    #    return (self._I(p, z) + self._S(p, z) * (1. - np.exp(-tau))) * p

@numba.cfunc(numba.f8(numba.i4,numba.types.CPointer(numba.f8)))
def get_Iemit(n, ptr):
    temp = numba.carray(ptr, n)
    
    vmax, vphot, t, vdet_min, vdet_max, tauref, ve = np.float32(temp[2]), np.float32(temp[3]), np.float32(temp[4]),\
          np.float32(temp[5]), np.float32(temp[6]), np.float32(temp[7]), np.float32(temp[8])    
    return _Iemit(np.float32(temp[0]), np.float32(temp[1]), vmax, vphot, t, vdet_min, vdet_max, tauref, ve)


class PcygniCalculator(object):
    """
    Calculator for P-Cygni line profiles emerging from homologously expanding
    supernova ejecta flows using the Elementary Supernova model and
    prescriptions outlined in Jeffery & Branch 1990.

    This calculator heavily relies on the impact geometry (see Fig. 1 in
    Jeffery & Branch 1990) in which the z-axis points towards the observer and
    the impact parameters p is defined perpendicular to that. The connection to
    spherical symmetry is achieved by mu=z/r, r=sqrt(z**2 + p**2).

    The only routines which should be used by the user are the ones without an
    underscore, i.e.:
    * calc_profile_Fnu
    * calc_profile_Flam
    * show_line_profile
    """
    def __init__(self, t=3000 * units.s, vmax=0.01 * csts.c,
                 vphot=0.001 * csts.c, tauref=1, vref=5e7 * units.cm/units.s,
                 ve=5e7 * units.cm/units.s, lam0=1215.7 * units.AA,
                 vdet_min=None, vdet_max=None):
        """
        Parameters
        ----------
        t : scalar astropy.units.Quantity
            time since explosion; together with the photospheric and maximum
            velocity this sets the length scale of the ejecta (default 3000 s)
        vmax : scalar astropy.units.Quantity
            maximum ejecta velocity; with the time since explosion, this sets
            the outer radius of the ejecta (default 1 per cent speed of light)
        vphot : scalar astropy.units.Quantity
            photospheric velocity; with the time since explosion, this sets the
            radius of the photosphere, i.e. of the inner boundary (default 0.1
            per cent of speed of light)
        tauref : float
            line optical depth at a reference velocity (vref) in the ejecta;
            this sets the strength of the line transition (default 1)
        vref : scalar astropy.units.Quantity
            reference velocity; needed in the assumed density stratification
            and sets the ejecta location where the reference line optical depth
            is measured (default 5e7 cm/s)
        ve : scalar astropy.units.Quantity
            second parameter used in the assumed density stratification
            (defautl 5e7 cm/s)
        lam0 : scalar astropy.units.Quantity
            rest frame wavelength of the line transition (default 1215.7 A)
        vdet_min : None or scalar astropy.units.Quantity
            lower/inner location of the line formation region; enables
            detachment of line formation region; if None, will be set to vphot
            (default None)
        vdet_max : None or scalar astropy.units.Quantity
            upper/outer location of the line formation region; enables
            detachment of line formation region; if None, will be set to vmax
            (default None)
        """

        # ensure that the calculator works with the correct units
        self._t = t.to("s").value
        self._vmax = vmax.to("cm/s").value
        self._vphot = vphot.to("cm/s").value
        self._ve = ve.to("cm/s").value
        self._vref = vref.to("cm/s").value

        # spatial extent of the ejecta
        self._rmax = self._t * self._vmax
        self._rmin = self._t * self._vphot
        self._zmax = self._rmax

        # CMF natural wavelength and frequency of the line
        self._lam0 = lam0.to("cm").value
        self._nu0 = csts.c.cgs.value / self._lam0

        # determine the maximum width of the profile
        dlambda = self._lam0 / self._t * self._zmax / csts.c.cgs.value

        # determine the wavelength/frequency range over which the profile will
        # be calculated (5% more than maximum Doppler shift on both ends)
        self._lam_min = self._lam0 - 1.05 * dlambda
        self._lam_max = self._lam0 + 1.05 * dlambda
        self._nu_min = csts.c.cgs.value / self._lam_max
        self._nu_max = csts.c.cgs.value / self._lam_min

        self._tauref = tauref
        self._Ip = 1

        if vdet_min is None:
            vdet_min = self.vphot
        else:
            vdet_min = vdet_min.to("cm/s").value
        if vdet_max is None:
            vdet_max = self.vmax
        else:
            vdet_max = vdet_max.to("cm/s").value

        self._vdet_min = vdet_min
        self._vdet_max = vdet_max

        t0 = t.to("s").value
        ve0 = ve.to("cm/s").value
        vmax0 = vmax.to("cm/s").value
        vphot0 = vphot.to("cm/s").value
        #vdm0 = vmax.to("cm/s").value
        self.args = (vmax0, vphot0, t0, vdet_min, vdet_max, tauref, ve0)
        self._Iemit = scipy.LowLevelCallable(get_Iemit.ctypes)
    # Using properties allows the parameters to be crudely "hidden" from the
    # user; thus he is less likely to change them after initialization
    @property
    def t(self):
        """time since explosion in s"""
        return self._t

    @property
    def vmax(self):
        """maximum ejecta velocity in cm/s"""
        return self._vmax

    @property
    def vphot(self):
        """photospheric velocity in cm/s"""
        return self._vphot

    @property
    def ve(self):
        """velocity scale in density profile in cm/s"""
        return self._ve

    @property
    def vref(self):
        """reference velocity in cm/s"""
        return self._vref

    @property
    def vdet_min(self):
        """inner location of line-formation region in cm/s"""
        return self._vdet_min

    @property
    def vdet_max(self):
        """outer location of line-formation region in cm/s"""
        return self._vdet_max

    @property
    def rmax(self):
        """outer ejecta radius in cm"""
        return self._rmax

    @property
    def rmin(self):
        """photospheric radius in cm"""
        return self._rmin

    @property
    def zmax(self):
        """maximum z-coordinate in ejecta in cm (corresponds to rmax)"""
        return self._zmax

    @property
    def lam0(self):
        """CMF natural wavelength of line transition in cm"""
        return self._lam0

    @property
    def nu0(self):
        """CMF natural frequency of line transition in Hz"""
        return self._nu0

    @property
    def lam_min(self):
        """minimum wavelength for line profile calculation in cm"""
        return self._lam_min

    @property
    def lam_max(self):
        """maximum wavelength for line profile calculation in cm"""
        return self._lam_max

    @property
    def nu_min(self):
        """minimum frequency for line profile calculation in Hz"""
        return self._nu_min

    @property
    def nu_max(self):
        """maximum frequency for line profile calculation in Hz"""
        return self._nu_max

    @property
    def Ip(self):
        """photospheric continuum specific intensity in arbitrary units"""
        return self._Ip

    @property
    def tauref(self):
        """reference line optical depth"""
        return self._tauref

    def _calc_z(self, nu):
        """
        Calculate location (in terms of z) of resonance plane for photon
        emitted by the photosphere with frequency nu

        Parameters
        ----------
        nu : float
            photospheric photon frequency

        Returns
        -------
        z : float
            z coordinate of resonance plane
        """

        return csts.c.cgs.value * self.t * (1. - self.nu0 / nu)


    def _calc_line_flux(self, nu, mode="both"):
        """
        Calculate the emergent flux at LF frequency nu

        Parameters
        ----------
        nu : float
            lab frame frequency at which the line flux is to be calculated
        mode : str
            identifies the included interaction channels; see self.Iemit
            (default 'both')

        Returns
        -------
        Fnu : float
            emergent flux F_nu
        """

        #z = self._calc_z(nu)
        delta = nu/self.nu0
        
        pmax = self.rmax

        # integration over impact parameter p
        Fnu = 2. * np.pi * integ.quad(self._Iemit, 0, pmax, args=(delta, *self.args), epsabs=1)[0]
        return Fnu

    def _calc_line_profile_base(self, nu_min, nu_max, npoints=100,
                                mode="both"):
        """
        Calculate the full line profile between the limits nu_min and nu_max in
        terms of F_nu.

        Parameters
        ----------
        nu_min : float
            lower frequency limit
        nu_max : float
            upper frequency limit
        npoints : int
            number of points of the equidistant frequency grid (default 100)
        mode : str
            identifier setting the interaction mode, see self.Iemit
            (default 'both')

        Returns
        -------
        nu : np.ndarray
            frequency grid
        Fnu : np.ndarray
            emitted flux F_nu
        """

        nu = np.linspace(nu_min, nu_max, npoints)

        Fnu = []
        for nui in nu:
            Fnu.append(self._calc_line_flux(nui, mode=mode))

        #with Pool(4) as p:     
        #    Fnu = p.map(self._calc_line_flux, nu)
        
        return nu * units.Hz, np.array(Fnu)

    def calc_profile_Fnu(self, npoints=100, mode="both"):
        """Calculate normalized line profile in terms of F_nu

        Parameters
        ----------
        npoints : int
            number of points of the equidistant frequency grid (default 100)
        mode : str
            identifier setting the interaction mode, see self.Iemit
            (default 'both')

        Returns
        -------
        nu : np.ndarray
            frequency grid
        Fnu_normed : np.ndarray
            emitted flux F_nu, normalized to the emitted photospheric continuum
            flux
        """

        nu, Fnu = self._calc_line_profile_base(self.nu_min, self.nu_max,
                                               npoints=npoints, mode=mode)

        Fnu_normed = Fnu / Fnu[0]
        return nu, Fnu_normed

    def calc_profile_Flam(self, npoints=100, mode="both"):
        """Calculate normalized line profile in terms of F_lambda

        Parameters
        ----------
        npoints : int
            number of points in the wavelength grid. NOTE even though a
            F_lam(lam) is calculated the underlying wavelength grid is chosen
            such that it is equidistant in nu-space (since the actual
            integration happens in terms of F_nu(nu))
            (default 100)
        mode : str
            identifier setting the interaction mode, see self.Iemit
            (default 'both')

        Returns
        -------
        lam : np.ndarray
            wavelength grid
        Flambda_normed : np.ndarray
            emitted flux F_lambda, normalized to the emitted photospheric
            continuum flux
        """

        nu, Fnu = self._calc_line_profile_base(self.nu_min, self.nu_max,
                                               npoints=npoints, mode=mode)
        lam = nu.to("AA", equivalencies=units.spectral())[::-1]
        cont = (Fnu[0] * np.ones(len(Fnu)) * nu.to("Hz").value**2 /
                csts.c.cgs.value)
        F_lambda_normed = (Fnu * nu.to("Hz").value**2 /
                           csts.c.cgs.value / cont)[::-1]

        return lam, F_lambda_normed

    def show_line_profile(self, npoints=100, include_abs=True,
                          include_emit=True, vs_nu=False):
        """
        Visualise Line Profile

        The P-Cygni line profile will always be displayed. The pure absorption
        and emission components can be included in the plot as well. The flux
        (will always be be F_nu) may be plotted against frequency or
        wavelength.

        Arguments:
        nu_min  -- lower frequency limit
        nu_max  -- upper frequency limit

        Keyword arguments:
        npoints -- number of points of the frequency grid (default 100)
        include_abs  -- if True, the pure absorption flux will be included and
                        shown as a separate line (default True)
        include_emit -- if True, the pure emission flux will be included and
                        shown as a separate line (default True)
        vs_nu -- if True the quantities will be shown against frequency,
                 otherwise against wavelength (default True)

        Returns:
        fig -- figure instance containing plot
        """

        if vs_nu:
            x, y = self.calc_profile_Fnu(npoints=npoints, mode="both")
            x = x.to("Hz")
            if include_abs:
                yabs = self.calc_profile_Fnu(npoints=npoints, mode="abs")[-1]
            if include_emit:
                yemit = self.calc_profile_Fnu(npoints=npoints, mode="emit")[-1]
        else:
            x, y = self.calc_profile_Flam(npoints=npoints, mode="both")
            x = x.to("AA")
            if include_abs:
                yabs = self.calc_profile_Flam(npoints=npoints, mode="abs")[-1]
            if include_emit:
                yemit = self.calc_profile_Flam(
                    npoints=npoints, mode="emit")[-1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)

        if include_abs:
            ax.plot(x, yabs, color="grey", ls="dashed",
                    label="absorption component")
        if include_emit:
            ax.plot(x, yemit, color="grey", ls="dotted",
                    label="emission component")

        ax.plot(x, y, ls="solid",
                label="emergent line profile")
        ax.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=2,
                  mode="expand", borderaxespad=0.)

        if vs_nu:
            ax.set_xlabel(r"$\nu$ [Hz]")
            ax.set_ylabel(r"$F_{\nu}/F_{\nu}^{\mathrm{phot}}$")
        else:
            ax.set_xlabel(r"$\lambda$ [$\AA$]")
            ax.set_ylabel(r"$F_{\lambda}/F_{\lambda}^{\mathrm{phot}}$")

        ax.set_xlim([np.min(x.value), np.max(x.value)])

        return fig


def example():
    """a simple example illustrating the use of the line profile calculator"""

    prof_calc = PcygniCalculator(t=3000 * units.s, vmax=0.01 * csts.c,
                                 vphot=0.001 * csts.c, tauref=1, vref=5e7 *
                                 units.cm/units.s, ve=5e7 * units.cm/units.s,
                                 lam0=1215.7 * units.AA)

    prof_calc.show_line_profile(npoints=100)
    plt.show()

if __name__ == "__main__":

    example()

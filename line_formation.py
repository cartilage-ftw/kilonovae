from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c, h, k_B
from astropy.modeling.physical_models import BlackBody
from astropy.units import Quantity
from scipy.integrate import quad

"""
Does the PCygni line formation while taking into account the level populations,
and optical depths not just at the photosphere but at each point on a grid, taken in the velocity space
of the ejecta profile

Author: Aayush
(so you know whom to blame for mistakes)
"""

@dataclass
class LineTransition:
    """
    Holds the line transition information, along with the level populations and optical depth
    provided from the NLTE calculation.

    When tau is needed for a specific r (or v), it returns the value by interpolating.
    """
    wavelength: Quantity
    tau_grid: np.ndarray # shape same as len(velocity_grid)
    velocity_grid: np.ndarray # in units of 'c'
    g_upper: int
    g_lower: int
    n_upper: np.ndarray # level population of upper level
    n_lower: np.ndarray # shape same as len(velocith_grid)

    def __post_init__(self):
        self.frequency = self.wavelength.to("Hz", equivalencies=u.spectral())
        self.n_u = lambda v: np.interp(v/c, self.velocity_grid, self.n_upper)
        self.n_l = lambda v: np.interp(v/c, self.velocity_grid, self.n_lower)
    
    
    def source_function(self, v):
        nu = self.frequency
        _S = (2 * h * nu**3 / c**2) / (self.g_upper * self.n_l(v) / (self.g_lower * self.n_u(v)) - 1)
        #print("Unit of the source function is", _S.si.unit)
        return _S / u.sr
    
    def tau(self, v):
        return np.interp(v/c, self.velocity_grid, self.tau_grid)


def calc_r(p: float, z: float):
    return np.sqrt(p**2 + z**2)


def calc_z(p: float, t: float, delta: np.array):
    """
    Location of the Sobolev resonance plane. A specific intensity beam I_\nu interacts
    with a resonance transition when the red/blueshift of the photon in the local frame of the atoms
    matches the rest-frame transition frequency. This happens only at specific points in the ejecta
    called Sobolev points/planes.

    In impact geometry, this is a plane of constant 'z' along the observer line of sight
    """
    # the location of the resonance plane
    # includes higher order relativistic corrections as per Hutchsemekers & Surdej
    # I copied the expression from Albert's update
    z = c*t* (delta**2/(1+delta**2) - delta**2/(1+delta**2) * (1+(1+delta**2)/delta**4*( (1-delta**2-(p/c/t)**2)))**(1/2))
    return z.cgs


@dataclass
class Photosphere:
    """
    I defined this datastructure because it turned out to be convenient for not having to pass these as parameters in the function
    all the fucking time. Now I understand why other people did it this way in their version
    """
    v_phot: Quantity
    v_max: Quantity
    t_d: Quantity # time since explosion, in days
    continuum: BlackBody# Doesn't need to be blackbody tbh. Note that however, for modelling nebular phase
                        # what instead we need to worry about are more general forms of the source function/emissivity
                        # including the non-scattering contribution G(r), etc.
    line_list: list[LineTransition]

    def __post_init__(self):
        self.r_max = (self.v_max * self.t_d).to("cm")
        self.r_min = (self.v_phot * self.t_d).to("cm")
        self.line_wavelengths = np.array([line.wavelength.to("AA").value for line in self.line_list]) * u.AA
        self.rest_frequencies = self.line_wavelengths.to("Hz", equivalencies=u.spectral())
    
    def W(self, v):
        """
        The geometric dilution factor.
        """
        return (1 - np.sqrt(1 - (self.v_phot/v)**2)) / 2
    
    
    def I(self, p, z, nu):
        """
        The incident specific intensity beam
        """
        r = np.sqrt(p**2 + z**2) 
        return np.where(p < self.r_min, self.continuum(nu), 0) * c / nu**2
        #    return 0
        #return self.continuum(nu)

    
    def S(self, p, z, nu, line: LineTransition, mode='level-populations'):
        """
        The source function, defined as the ratio of the emissivity to the absorption coefficient
        """
        r = calc_r(p, z)
        v = r/self.t_d
        # in the occulted region, or outside the line forming region
        outside_photosphere = (r < self.r_min) | (r > self.r_max)
        occulted_region = (r < 0) & (p < self.r_min)
        
        return np.where(occulted_region | outside_photosphere, 0., line.source_function(v))

        #if r < self.r_min or r > self.r_max:
        #    return 0
        #elif z < 0 and p < self.r_min:
        #    return 0
        #else:
        #    return line.source_function(v) #_S 
        #_S = line.source_function(v)
        #if mode=='two-level-approx':
        #    _S = W(v) * I(p, z, nu)

    
    def I_emit(self, p: float, nu: float, delta_arr: np.array):
        # the line summation has to be done in the minus \hat{n} direction (i.e. going inwards
        # instead of outwards from observer's line of sight); see eqn (22) of Jeffrey & Branch (1990)
        # to do the ordering, you have to know where the Sobolev resonance plane lies for each particular line
        z = calc_z(p, self.t_d, delta_arr)
        # NOTE: Remember that in cgs units these quantities have values of the order of 10^15
        #print("Unit of z", z)
        #print("Unit of p", p)
        v = calc_r(p, z)/self.t_d
        order = np.argsort(z)[::-1]
        line_transitions = np.array(self.line_list)
        tau_arr = []
        S_i = []
        for line in line_transitions[order]:
            tau_arr.append(line.tau(v))
            S_i.append(self.S(p,z,nu, line))
        tau_arr = np.array(tau_arr, dtype=float)

        
        #print("Unit of the source function", S_i[0].unit.decompose())


        I_abs = (self.I(p, z, nu)*np.exp(-tau_arr)).sum(axis=0)
        I_scat = []
        for i in range(len(tau_arr)):
            I_scat.append(
                    S_i[i] * (1-np.exp(-tau_arr[i])) * np.exp(-tau_arr[:i-1].sum())
                        )

        #print("Unit of the scattering term", I_scat[-1].decompose())
        I_scat = np.array(I_scat) * S_i[0].unit #/ u.sr
        #print("Unit of it now", I_scat[-1].unit.decompose())
        #print("unit of I_abs", I_abs[0].unit.decompose())
        I_final = (I_abs + I_scat).sum()
        #print("I_emit is", I_final)
        return p * I_final


    def I_emit_wrapper(self, p: float, nu: float, delta_arr: np.array):
        p *= u.cm
        nu *= u.Hz
        _Iemit = self.I_emit(p, nu, delta_arr)
        #print(f"Unit of _Iemit (lambda={nu.to('nm', equivalencies=u.spectral()):.2f})", _Iemit.unit)
        return _Iemit.cgs.value
    
    
    def calc_spectrum(self, start_wav=3_500*u.AA, end_wav=15_500*u.AA):
        """
        I was thinking, if I pass v_phot and v_max here that can be used for a fitting routine
        it can be used to update v_phot and v_max set in the Photosphere object
        """ 
        n_points = 100

        # It may happen that the user specified wavelength range actually does not include the bluest or reddest part of the line
        lambda_min =(1-self.v_phot/c)*np.min(self.line_wavelengths.to("AA").value) * 0.95
        lambda_max = (1+self.v_max/c)*np.max(self.line_wavelengths.to("AA").value) * 1.05
        # in that case, just do the calculation in regions outside it.
        lambda_min = np.min(start_wav)
        lambda_max = np.max(end_wav)
        
        # In a lot of the spectrum, there may not be any opacity to cause absorption/emission
        # in those parts, just return the continuum. Calculate only in relevant areas.

        nu_min = (c/lambda_max).to('Hz', equivalencies=u.spectral()).value
        nu_max = (c/lambda_min).to('Hz', equivalencies=u.spectral()).value

        nu_grid = np.linspace(nu_min, nu_max, n_points)
        
        Fnu_list = []
        for nu in nu_grid:
            delta = nu/self.rest_frequencies.value
            print(f"Evaluating flux for lambda={nu.to('nm', equivalencies=u.spectral()):.2f}")
            F_nu = 2*np.pi*quad(self.I_emit_wrapper, 0, self.r_max.cgs.value, args=(nu, delta,))[0]
            Fnu_list.append(F_nu)

        wavelength_grid = (c/(nu_grid*u.Hz)).to("AA")
        F_lambda = (np.array(Fnu_list) * nu**2 / c)[::-1]
        # NOTE: F_nu and F_lambda are not the same. F_lambda = F_nu * (c/lambda^2)
        # F_lambda = F_nu * (c/lambda^2)
        # do that conversion first, before returning
        return wavelength_grid, F_lambda


if __name__ == "__main__":
    # some sanity checking plots
    print("Nothing crashed! Can we celebrate that?")



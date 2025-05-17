from astropy import units as u
from astropy.constants import c, h, k_B
from astropy.modeling.physical_models import BlackBody
from astropy.units import Quantity
from numba import njit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp

"""
Does the PCygni line formation while taking into account the level populations,
and optical depths not just at the photosphere but at each point on a grid, taken in the velocity space
of the ejecta profile

Author: Aayush
(so you know whom to blame for mistakes)
"""

num_cores = mp.cpu_count()
print("Number of CPU cores available", num_cores)

# I don't like cgs units, but we're doing astronomy so I have to live with it
c = c.cgs.value
h = h.cgs.value
k_B = k_B.cgs.value


@njit(fastmath=True)
def fast_interpolate(x_new, v_grid, y_grid):
    """
    A fast interpolation function to be used in the numba-compiled code
    """
    return np.interp(x_new, v_grid, y_grid)


@njit(fastmath=True)
def source_function(v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0):
    n_u = np.interp(v / c, v_grid, n_u_grid)
    n_l = np.interp(v / c, v_grid, n_l_grid)
    return (2 * h * nu_0**3 / c**2) / (g_u * n_l / (g_l * n_u) - 1)


@njit(fastmath=True)
def S(p, z, r, r_min, r_max, v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0):

    outside_photosphere = (r < r_min) | (r > r_max)
    occulted_region = (z < 0) & (p < r_min)
    return np.where(outside_photosphere | occulted_region,
                    1e-20,
                    source_function(v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0))

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
        self.nu_0 = self.wavelength.to("Hz", equivalencies=u.spectral())#.value
        self.wavelength = self.wavelength.to("cm")#.value
        # compile an interpolation grid for tau, n_upper and n_lower
        #self._n_u_interp = lambda v: fast_interpolate(v, self.velocity_grid, self.n_upper) #interp1d(self.velocity_grid, self.n_upper, kind='linear', bounds_error=False, fill_value='extrapolate')
        #self._n_l_interp = lambda#interp1d(self.velocity_grid, self.n_lower, kind='linear', bounds_error=False, fill_value='extrapolate')
        #self._tau_interp = #interp1d(self.velocity_grid, self.tau_grid, kind='linear', bounds_error=False, fill_value='extrapolate')
        # later, one can sample from these while integrating.
        self.n_u = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_upper)#self._n_u_interp(v/c)
        self.n_l = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_lower)#self._n_l_interp(v/c)
        self._tau_interp = lambda v: fast_interpolate(v/c, self.velocity_grid, self.tau_grid)
        #self.tau = lambda v: self._tau_interp(v/c)
    
    #def source_function(self, v):
        #_S = 
        #print("Unit of the source function is", _S.si.unit)
    #    return (2 * h * self.nu_0**3 / c**2) / (self.g_upper * self.n_l(v) / (self.g_lower * self.n_u(v)) - 1)#_S# / u.sr
    
@njit
def calc_r(p: float, z: float):
    return np.sqrt(p**2 + z**2)

@njit
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
    return z#.cgs.value



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
        self.r_max = (self.v_max * self.t_d).to("cm").value
        self.r_min = (self.v_phot * self.t_d).to("cm").value
        self.line_wavelengths = np.array([line.wavelength.to("cm").value for line in self.line_list]) * u.cm
        self.rest_frequencies = self.line_wavelengths.to("Hz", equivalencies=u.spectral()).value
        self.v_phot = self.v_phot.to("cm/s").value
        self.v_max = self.v_max.to("cm/s").value
        self.t_d = self.t_d.to("s").value

    def tau(self, line, v, r, z):
        # including the relativistic correction for tau from Hutsemekers & Surdej (1990)
        mu = z / r
        beta = v/c
        corr = abs( (1-mu*beta)**2 / ( (1-beta)*(mu*(mu-beta) + (1- mu**2)*(1-beta**2)) ) )
        tau_ = np.where( 
                        (v >= self.v_phot) & (v < self.v_max),
                        corr * line._tau_interp((v/c)),
                        1e-20
                    )
        return tau_

    
    def W(self, v):
        """
        The geometric dilution factor.
        """
        return (1 - np.sqrt(1 - (self.v_phot/v)**2)) / 2
    
    
    def I(self, p, z, nu, continuum):
        """
        The incident specific intensity beam
        """
        return np.where(p < self.r_min, continuum, 0) * c / nu**2
        #    return 0
        #return self.continuum(nu)

    '''def S(self, p, z, r, v, nu, line: LineTransition, mode='level-populations'):
        """
        The source function, defined as the ratio of the emissivity to the absorption coefficient
        """
        # in the occulted region, or outside the line forming region
        
        
        return np.where(occulted_region | outside_photosphere,
                        0.,
                        source_function(v,
                                        line.g_upper,
                                        line.g_lower, 
                                        line.n_upper,
                                        line.n_lower,
                                        line.velocity_grid,
                                        line.nu_0))'''

        #if r < self.r_min or r > self.r_max:
        #    return 0
        #elif z < 0 and p < self.r_min:
        #    return 0
        #else:
        #    return line.source_function(v) #_S 
        #_S = line.source_function(v)
        #if mode=='two-level-approx':
        #    _S = W(v) * I(p, z, nu)

    def I_emit(self, p: float, nu: float, delta_arr: np.array, continuum: float):
        # the line summation has to be done in the minus \hat{n} direction (i.e. going inwards
        # instead of outwards from observer's line of sight); see eqn (22) of Jeffrey & Branch (1990)
        # to do the ordering, you have to know where the Sobolev resonance plane lies for each particular line
        t_e = time.time()
        t_ei = t_e
        z = calc_z(p, self.t_d, delta_arr)
        # NOTE: Remember that in cgs units these quantities have values of the order of 10^15
        #print("Unit of z", z)
        #print("Unit of p", p)
        r = calc_r(p, z)
        v = r/self.t_d#)#.to('m/s')
        #print("Calculating r,z,v, took", time.time() - t_e)
        t_e = time.time()
        #print("value of 'v'", v.to('m/s'))
        order = np.argsort(z)[::-1]
        line_transitions = np.array(self.line_list)
        #print("Sorting took", time.time() - t_e)
        t_e = time.time()
        tau_arr = []
        S_i = lambda line, p, z, r, v: S(p, z, r, self.r_min, self.r_max, v, line.g_upper, line.g_lower, line.n_upper, line.n_lower, line.velocity_grid, line.nu_0)
        S_ = []
        for line in line_transitions[order]:
            tau_arr.append(self.tau(line, v, r, z))
            S_.append(S_i(line, p,z,r,v))
        tau_arr = np.array(tau_arr, dtype=float)
        #print("Interpolating tau took", time.time() - t_e)
        t_e = time.time()

        I_abs = np.sum(self.I(p, z, nu, continuum)*np.exp(-tau_arr))
        #print("Calculating I_abs took", time.time() - t_e)
        t_e = time.time()

        I_line_emit = []
        for i in range(len(tau_arr)):
            I_line_emit.append(
                    S_[i] * (1-np.exp(-tau_arr[i])) * np.exp(-tau_arr[:i-1].sum())
                        )
        #print("Calculating I_line_emit took", time.time() - t_e)
        #print("One step of the integration took", time.time() - t_ei)
        return p * (I_abs + np.sum(I_line_emit))
        
    
    def get_line_mask(self, nu_grid):
        """
        There are parts of the spectrum where there are no lines
        In those parts, the flux is just the continuum.
        Use this function to get the mask of the line regions and avoid
        calculating I_emit in those regions.
        """
         # Compute the influence range of each line
        nu_min_arr = self.rest_frequencies / (1 + self.v_max / c)
        nu_max_arr = self.rest_frequencies / (1 - self.v_max / c)

        #fig, ax = plt.subplots()
        mask = np.zeros_like(nu_grid, dtype=bool)
        #offset = 0
        #ax.scatter(c*1e7/nu_grid[mask], (np.ones_like(nu_grid) + offset)[mask], s=1, alpha=0.2)
        for i, (nu_min, nu_max) in enumerate(zip(nu_min_arr, nu_max_arr)):
            mask |= (nu_grid >= nu_min) & (nu_grid <= nu_max)
            #offset += 1
            #ax.scatter(c*1e7/nu_grid[mask], (np.ones_like(nu_grid) + offset)[mask], s=1, alpha=0.2)
            #ax.axvline(x=c*1e7/nu_min, ymin=0 + offset/10, ymax=0.1 + offset/10, linestyle='--')
            #ax.axvline(x=c*1e7/nu_max, ymin=0 + offset/10, ymax=0.1 + offset/10, linestyle='--')
            #ax.text(c*1e7/self.rest_frequencies[i], 1 + offset, va='center',
            #         s=f"{c*1e7/self.rest_frequencies[i]:.2f}", fontsize=11)
        #plt.show()
        return ~mask
    

    def calc_spectrum(self, start_wav=3_500*u.AA, end_wav=22_500*u.AA, n_points=200):
        """
        I was thinking, if I pass v_phot and v_max here that can be used for a fitting routine
        it can be used to update v_phot and v_max set in the Photosphere object
        """

        # It may happen that the user specified wavelength range actually does not include the bluest or reddest part of the line
        lambda_min =(1-self.v_phot/c)*np.min(self.line_wavelengths.to("cm").value) * 0.95
        lambda_max = (1+self.v_max/c)*np.max(self.line_wavelengths.to("cm").value) * 1.05
        # in that case, just do the calculation in regions outside it.
        lambda_min = np.min([start_wav.to("cm").value, lambda_min])
        lambda_max = np.max([end_wav.to("cm").value, lambda_max])
        
        # In a lot of the spectrum, there may not be any opacity to cause absorption/emission
        # in those parts, just return the continuum. Calculate only in relevant areas.

        nu_min = (c/lambda_max)#.to('Hz', equivalencies=u.spectral()).value
        nu_max = (c/lambda_min)#.to('Hz', equivalencies=u.spectral()).value

        nu_grid = np.linspace(nu_min, nu_max, n_points)
        lamb_grid = 1e7*c/nu_grid
        Fnu_list = []
        # pre-compute the continuum flux (Planck function)
        B_nu_grid = self.continuum(nu_grid * u.Hz).cgs.value

        line_mask = self.get_line_mask(nu_grid)
        # now, calculate the line features
        for i, nu in enumerate(nu_grid):
            if line_mask[i]:
                # if the line is not in the line formation region, just return the continuum
                Fnu_list.append(B_nu_grid[i])
                #print("Skipping line formation for lambda", lamb_grid[i])
                continue
            delta = nu/self.rest_frequencies#.value
            print(f"Evaluating flux for lambda={(c*1e7/nu)}")
            t_i = time.time()
            p_grid = np.linspace(0, self.r_max, 200)
            I_vals =  np.array([
                self.I_emit(p, nu, delta, B_nu_grid[i]) for p in p_grid
            ])
            F_nu = 2* np.pi * np.trapz(I_vals, p_grid)
            #F_nu = 2*np.pi*quad(self.I_emit_wrapper, 0, self.r_max.cgs.value, args=(nu, delta,))[0]
            print(f"Time taken: {(time.time() - t_i):.3f} seconds")
            Fnu_list.append(F_nu)

        wavelength_grid = 1e8*(c/(nu_grid))
        F_lambda = (np.array(Fnu_list) * nu**2 / c)[::-1]
        # NOTE: F_nu and F_lambda are not the same. F_lambda = F_nu * (c/lambda^2)
        # F_lambda = F_nu * (c/lambda^2)
        # do that conversion first, before returning
        return wavelength_grid, F_lambda


if __name__ == "__main__":
    # some sanity checking plots
    print("Nothing crashed! Can we celebrate that?")



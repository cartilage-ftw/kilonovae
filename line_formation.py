from synphot.units import convert_flux
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
import multiprocessing as mp
import time

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
def fast_interpolate(x_new: float, v_grid: np.array, y_grid: np.array):
    """
    A fast interpolation function to be used in the numba-compiled code
    """
    return np.interp(x_new, v_grid, y_grid)#np.where(
                     #)


@njit(fastmath=True)
def W(r, r_min):
    """
    The geometric dilution factor.
    """
    return (1 - np.sqrt(1 - (r_min/r)**2)) / 2


@njit(fastmath=True)
def source_function(v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0):
    n_u = np.interp(v / c, v_grid, n_u_grid)
    n_l = np.interp(v / c, v_grid, n_l_grid)
    return (2 * h * nu_0**3 / c**2) / (g_u * n_l / (g_l * n_u) - 1)


@njit(fastmath=True)
def S(p, z, r, r_min, r_max, v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0, continuum):

    outside_photosphere = (r < r_min) | (r > r_max)
    occulted_region = (z < 0) & (p < r_min)
    return np.where(outside_photosphere | occulted_region,
                    1e-10,
                    W(r, r_min)*continuum#*4e-19
                    #source_function(v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0)
                    )

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
        self._tau_interp = interp1d(self.velocity_grid, self.tau_grid, kind='linear', bounds_error=False, fill_value=1e-20)
        # later, one can sample from these while integrating.
        self.n_u = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_upper)#self._n_u_interp(v/c)
        self.n_l = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_lower)#self._n_l_interp(v/c)
        #self._tau_interp = lambda v: fast_interpolate(v/c, self.velocity_grid, self.tau_grid)
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
    z = c*t*(1-delta)#(delta**2/(1+delta**2) - delta**2/(1+delta**2) * (1+(1+delta**2)/delta**4*( (1-delta**2-(p/c/t)**2)))**(1/2))
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

        print("Initializing Photosphere for lines", self.line_wavelengths*1e7,"nm")
    def tau(self, line, p, z, r,v):
        # including the relativistic correction for tau from Hutsemekers & Surdej (1990)
        mu = z / r
        beta = v/c
        corr = abs( (1-mu*beta)**2/( (1-beta)*(mu*(mu-beta)+(1-mu**2)*(1-beta**2))) )
        #print("magnitude of the correction factor", corr)
        #indices = np.where((corr > 1.5) & (v >= self.v_phot) & (v < self.v_max))[0]
        #if np.any(indices):
            #print("Warning: The correction factor is too large", corr[indi], "for v", v[indices & reasonable_v]/c, "and z", z[indices& reasonable_v]/(c*self.t_d), "and r", r[indices& reasonable_v]/(c*self.t_d))
            #print("Warning: The correction factor is too large", corr[indices], "for v", v[indices]/c, "and z", z[indices]/(c*self.t_d), "and r", r[indices]/(c*self.t_d))
        outside_photosphere = (r <= self.r_min) | (r > self.r_max)
        occulted_region = (z < 0) & (p < self.r_min)
        if ~np.any(occulted_region | outside_photosphere):
            print("For line", line.wavelength*1e7, "r ", r/(c*self.t_d), "z", z/(c*self.t_d), "p", p/(c*self.t_d), "v", v/c)
            print("Occulted?", occulted_region, "Outside photosphere?", outside_photosphere)
        tau_ = np.where( 
                        outside_photosphere | occulted_region,
                        1e-15,
                        corr * line._tau_interp((v/c)),
                    )
        return tau_
    
    
    def I(self, p, continuum):
        """
        The incident specific intensity beam
        """
        return np.where(p < self.v_phot * self.t_d, continuum, 0.)# * c / nu**2
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

    def I_emit(self, p:float, nu:float, delta_arr: np.array, continuum: float):
        z_arr = calc_z(p, self.t_d, delta_arr)
        r_arr = calc_r(p, z_arr)
        v_arr = r_arr/self.t_d
        tau_i = []

        # For the scattering part, the line summation has to be done in the minus \hat{n} direction (i.e. going inwards
        # instead of outwards from observer's line of sight); see eqn (22) of Jeffrey & Branch (1990)
        # to do the ordering, you have to know where the Sobolev resonance plane lies for each particular line
        order = np.argsort(z_arr)[::-1]
        
        I_scat = 0.
        # a helper function, because the S function is a bit long
        S_i = lambda line, p, z, r, v: S(p, z, r, self.r_min, self.r_max, v, line.g_upper, line.g_lower, line.n_upper, line.n_lower, line.velocity_grid, line.nu_0, continuum)  
        
        for i, line in enumerate(np.array(self.line_list)[order]):
            tau_i.append(self.tau(line, p, z_arr[order][i], r_arr[order][i], v_arr[order][i]))
            I_scat += S_i(line, p, z_arr[order][i], r_arr[order][i], v_arr[order][i]) * (1-np.exp(-tau_i[i])) * np.exp(-np.sum(tau_i[:i-1]))
        
        return p * (self.I(p, continuum)*np.exp(-np.sum(tau_i)) + I_scat)
    
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
        lamb_grid = (1e7*c/nu_grid) * u.nm
        Fnu_list = []
        # pre-compute the continuum flux (Planck function)
        B_lambda_grid = self.continuum(lamb_grid)#.cgs.value
        #print("Amplitude of blackbody", self.continuum.scale)
        #exit()
        #print("The unit of B_lambda", B_lambda_grid.unit)

        B_nu_grid = convert_flux(lamb_grid, B_lambda_grid, out_flux_unit='erg / (Hz s cm2 sr)').cgs#.value
        Bnu_cgs_unit = B_nu_grid.unit
        B_nu_grid = B_nu_grid.value

        line_mask = self.get_line_mask(nu_grid)
        # now, calculate the line features

        for i, nu in enumerate(nu_grid):
            if line_mask[i]:
                Fnu_list.append(np.pi * self.r_min**2 * B_nu_grid[i])
                continue
            delta = self.rest_frequencies/nu#.value
            #print(f"Evaluating flux for lambda={(c*1e7/nu)}")
            t_i = time.time()
            p_grid = np.linspace(0, self.r_max, 200)
            
            I_vals =  np.array([
                self.I_emit(p, nu, delta, B_nu_grid[i]) for p in p_grid
            ])
            F_nu = 2* np.pi * np.trapz(I_vals, p_grid)

            '''fig, ax = plt.subplots()
            ax.plot(p_grid/(c*self.t_d), I_vals, label='I_emit')
            ax.axvline(x=self.r_min/(c*self.t_d), color='red', linestyle='--', label='r_min')
            ax.axvline(x=self.r_max/(c*self.t_d), color='green', linestyle='--', label='r_max')
            #ax.plot(p_grid, B_nu_grid[i]*2*np.pi*self.r_min, label=r'$B_{\nu}$')
            #ax.plot(p_grid, F_nu)
            plt.show()'''
            #F_nu = 2*np.pi*quad(self.I_emit, 0, self.r_max, args=(nu, delta, B_nu_grid[i]))[0]

            print("Value of F_nu", F_nu)
            print(f"Time taken: {(time.time() - t_i):.3f} seconds")
            Fnu_list.append(F_nu)
        #ax.plot(z_grid/(self.t_d * c), tau_vals, label=self.line_wavelengths*1e7)
        #plt.show()
        F_lambda = convert_flux(nu_grid * u.Hz, Fnu_list * Bnu_cgs_unit * u.rad**2, out_flux_unit='1e20 erg / (s AA cm2)')# * u.sr
        wavelength_grid = (lamb_grid).to("AA").value
        #print("Order of magnitude of F_lambda", np.mean(F_lambda))
        return wavelength_grid, F_lambda.value/(np.pi*self.r_min**2)


if __name__ == "__main__":
    # some sanity checking plots
    print("Nothing crashed! Can we celebrate that?")



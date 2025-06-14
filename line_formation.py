import multiprocessing as mp
import time
from dataclasses import dataclass
from functools import lru_cache, partial

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c, h, k_B
from astropy.modeling.physical_models import BlackBody
from astropy.units import Quantity
from numba import njit
from pathos.multiprocessing import ProcessingPool
from scipy.integrate import quad
from scipy.interpolate import interp1d
from synphot.units import convert_flux
from matplotlib.colors import ListedColormap

"""
Does the PCygni line formation while taking into account the level populations,
and optical depths not just at the photosphere but at each point on a grid, taken in the velocity space
of the ejecta profile

Author: Aayush
(so you know whom to blame for mistakes)
"""

num_cores = mp.cpu_count()
#print("Number of CPU cores available", num_cores)

# I don't like cgs units, but we're doing astronomy so I have to live with it
c = c.cgs.value
h = h.cgs.value
k_B = k_B.cgs.value


@njit(fastmath=True)
def fast_interpolate(x_new: float, v_grid: np.array, y_grid: np.array):
    """
    A fast interpolation function to be used in the numba-compiled code
    """
    return np.interp(x_new, v_grid, y_grid)


@njit(fastmath=True)
def W(r, r_min):
    """
    The geometric dilution factor.
    """
    return (1 - np.sqrt(1 - (r_min/r)**2)) / 2


@njit(fastmath=True)
def source_function(v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0,
                    mode='nLTE'):
    """
    The source function, defined as the ratio of the emissivity to the absorption coefficient
    has the following form.
    """
    n_u = np.interp(v / c, v_grid, n_u_grid)
    n_l = np.interp(v / c, v_grid, n_l_grid)
    T = 4400
    if mode == 'LTE':
        return (2 * h * nu_0**3/c**2) / (np.exp(h*nu_0/(k_B * T)) - 1)
    return (2 * h * nu_0**3 / c**2) / (g_u * n_l / (g_l * n_u) - 1)


@njit(fastmath=True)
def S(p, z, r, r_min, r_max, v, sob_esc_prob, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0, continuum, mode='classic'):
    abs_only_mode = False
    outside_photosphere = (r < r_min) | (r > r_max)
    occulted_region = (np.abs(p) < r_min) & (z < 0) 
    if mode == 'level-pop':
        # the level populations are given from the NLTE calculation
        return np.where(outside_photosphere | occulted_region | abs_only_mode,
                        1e-10,
                        sob_esc_prob * source_function(v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0) / (4*np.pi)
                        )
    else:
        return np.where(outside_photosphere | occulted_region,
                    1e-10,
                    sob_esc_prob*W(r, r_min)*continuum
                    )

@dataclass
class LineTransition:
    """
    Holds the line transition information, along with the level populations and optical depth
    provided from the NLTE calculation.

    When tau is needed for a specific r (or v), it returns the value by interpolating.
    """
    wavelength: Quantity
    tau_grid_eq: np.ndarray # shape same as len(velocity_grid)
    tau_grid_polar: np.ndarray
    escape_prob_eq: np.ndarray
    escape_prob_polar: np.ndarray
    velocity_grid: np.ndarray # in units of 'c'
    g_upper: int
    g_lower: int
    n_upper: np.ndarray # level population of upper level
    n_lower: np.ndarray # shape same as len(velocith_grid)

    def __post_init__(self):
        self.nu_0 = self.wavelength.to("Hz", equivalencies=u.spectral())#.value
        self.wavelength = self.wavelength.to("cm")#.value
        # compile an interpolation grid for tau, n_upper and n_lower
        # later, one can sample from these while integrating.
        self.n_u = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_upper)
        self.n_l = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_lower)
        self._tau_interp_eq = lambda v: fast_interpolate(v/c, self.velocity_grid, self.tau_grid_eq)
        self._tau_interp_polar = lambda v: fast_interpolate(v/c, self.velocity_grid, self.tau_grid_polar)
        self.sob_esc_prob_eq = lambda v: fast_interpolate(v/c, self.velocity_grid, self.escape_prob_eq)
        self.sob_esc_prob_polar = lambda v: fast_interpolate(v/c, self.velocity_grid, self.escape_prob_polar)

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
    # includes higher order relativistic corrections as per Hutchsemekers & Surdej
    # I copied the expression from Albert's update and made it more readable ;)
    A = delta**2 / (1 + delta**2)
    B = (1 + (1 + delta**2) / delta**4 * ((1 - delta**2 - (p/(c*t))**2)))**(1/2)
    # NaN values may show up in 'B' if the square root factor in B becomes imaginary.
    # This is only going to be when delta is unphysically large (e.g. calculating if a 400nm photon
    # was redshifted to be in resonance with the 1 micron triplet; that's not happening even for v_max=c)
    B = np.where(np.isnan(B), 0., B)
    z = c * t * A * (1 - B)
    return z


@njit(fastmath=True)
def is_polar_ejecta(p, z, polar_opening_angle, observer_inclination_angle):
    """
    For implementing the two-component model, check if the point (p, z)
    is within the "polar" ejecta region or not.
    Also takes into account an observer's inclination angle
    """
    beta = observer_inclination_angle
    p_new = p * np.cos(beta) - z * np.sin(beta)
    z_new = p * np.sin(beta) + z * np.cos(beta)
    return np.where((np.abs((np.arctan(p_new/z_new))) < polar_opening_angle/2), 1, 0)


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
    polar_opening_angle: float = np.pi/4
    observer_angle: float = np.pi/2

    def __post_init__(self):
        self.r_max = (self.v_max * self.t_d).to("cm").value
        self.r_min = (self.v_phot * self.t_d).to("cm").value
        self.line_wavelengths = np.array([line.wavelength.to("cm").value for line in self.line_list]) * u.cm
        self.rest_frequencies = self.line_wavelengths.to("Hz", equivalencies=u.spectral()).value
        self.v_phot = self.v_phot.to("cm/s").value
        self.v_max = self.v_max.to("cm/s").value
        self.t_d = self.t_d.to("s").value
        print("Initializing Photosphere for lines", self.line_wavelengths*1e7,"nm")

    def visualize_polar_region(self):
        fig, ax = plt.subplots()
        v_phot_circle = plt.Circle((0, 0), self.v_phot/c, ec='w',ls='--', fill=False, alpha=1)
        v_max_circle = plt.Circle((0, 0), self.v_max/c, ec='w', ls=':', fill=False, alpha=1)
        p_list = np.linspace(-self.v_max/c, self.v_max/c, 500)
        z_list = np.linspace(-self.v_max/c, self.v_max/c, 500)#/(c*self.t_d)
        p_0 = np.zeros_like(p_list)# * self.r_max#/(c*self.t_d)
        z_grid, p_grid = np.meshgrid(z_list, p_list)#/(c*self.t_d)
        r_grid = calc_r(p_grid, z_grid)
        is_polar = is_polar_ejecta(p_grid, z_grid, self.polar_opening_angle, self.observer_angle)
        is_polar &= (r_grid) <= self.v_max/c
        
        trimmed_cmap = ListedColormap(plt.colormaps['coolwarm_r'](np.linspace(0.1,0.9,256)))
        
        pos = ax.imshow(is_polar, extent=(-self.v_max/c, self.v_max/c, -self.v_max/c, self.v_max/c), 
                        cmap=trimmed_cmap, origin='lower')
        fig.colorbar(pos, ax=ax,label='Polar Ejecta')
        ax.add_artist(v_phot_circle)
        ax.add_artist(v_max_circle)

        ax.plot(z_list, p_0, c='w', lw=1, ls='--')
        ax.text(0.7*self.v_phot/c, 0.02, s=r'$\to$', va='center', ha='center', color='w')
        ax.set_xlabel("z (c)")
        ax.set_ylabel("p (c)")
        plt.tight_layout()
        ax.set_aspect('equal')
        fig.savefig(f'polar_ejecta_{self.observer_angle:.1f}.png', dpi=300)
        plt.show()


    def tau(self, line, p, z, r,v, is_polar):
        # including the relativistic correction for tau from Hutsemekers & Surdej (1990)
        mu = z / r
        beta = v/c
        corr = abs( (1-mu*beta)**2/( (1-beta)*(mu*(mu-beta)+(1-mu**2)*(1-beta**2))) )
        
        # decide between polar and equatorial ejecta
        tau_ = np.where(is_polar,#is_polar_ejecta(p,z, self.polar_opening_angle, self.observer_angle),
                 corr*line._tau_interp_polar((v/c)),
                 corr*line._tau_interp_eq((v/c))
                 )

        outside_photosphere = (r <= self.r_min) | (r > self.r_max)
        occulted_region = (z < 0) & (np.abs(p) < self.r_min)
        return np.where( 
                        outside_photosphere | occulted_region,
                        1e-15,
                        tau_,
                    )
    
    
    def I(self, p, continuum):
        """
        The incident specific intensity beam
        """
        return np.where(np.abs(p) < self.v_phot * self.t_d, continuum, 0.)

    
    def I_emit(self, p:float, phi: float, nu:float, delta_arr: np.array, continuum: float):
        z_arr = calc_z(p, self.t_d, delta_arr)
        r_arr = calc_r(p, z_arr)
        v_arr = r_arr/self.t_d
        tau_i = []

        # project to sky-plane coordinates
        x = p * np.cos(phi)
        y = p * np.sin(phi)


        beta = self.observer_angle
        # tilt, for the polar ejecta cone by the observer angle
        x_rot = x * np.cos(beta) + z_arr*np.sin(beta)
        y_rot = y
        z_rot = -x * np.sin(beta) + z_arr * np.cos(beta)

        inside_polar = np.abs(z_rot / r_arr) >= np.cos(self.polar_opening_angle/2)

        # For the scattering part, the line summation has to be done in the minus \hat{n} direction (i.e. going inwards
        # from observer's line of sight to the center of the explosion, instead of inside to out); see eqn (22) of Jeffrey & Branch (1990)
        # to do the ordering, you have to know where the Sobolev resonance plane lies for each particular line;
        # and that's just 'z' in the impact geometry.
        order = np.argsort(z_arr)[::-1]
        
        I_scat = 0.
        # a helper function, because the S function is a bit long
        S_i = lambda line, p, z, r, v: S(p, z, r, self.r_min, self.r_max, v, line.sob_esc_prob_eq(v/c), line.g_upper, line.g_lower, line.n_upper, line.n_lower, line.velocity_grid, line.nu_0, continuum)  
        
        for i, line in enumerate(np.array(self.line_list)[order]):
            tau_i.append(self.tau(line, p, z_arr[order][i], r_arr[order][i], v_arr[order][i], inside_polar[order][i]))
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
    

    def calc_flux_at_nu(self, nu, delta, line_mask, B_nu, p_grid):
        if line_mask:
            return np.pi * self.r_min**2 * B_nu
        
        N_phi = 32 # number of points in the phi grid
        phi_grid = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
        dp = p_grid[1] - p_grid[0]
        dphi = phi_grid[1] - phi_grid[0]
        F_nu = 0.
        for p in p_grid:
            for phi in phi_grid:
                _Iemit = self.I_emit(p, phi, nu, delta, B_nu)
                F_nu += _Iemit * dp * dphi
        #t_i = time.time()
        #I_vals =  np.array([
        #    self.I_emit(p, nu, delta, B_nu) for p in p_grid
        #])
        #y_extent = 2 * np.sqrt( np.maximum(self.r_max**2 - p_grid**2, 0.0) )
        #F_nu = np.trapz(2*np.pi* I_vals, p_grid)
        #F_nu = 2*np.pi*quad(self.I_emit, 0, self.r_max, args=(nu, delta, B_nu_grid[i]))[0]
        #print("Value of F_nu", F_nu)
        return F_nu

    def calc_spectrum(self, start_wav=2350*u.AA, end_wav=24_500*u.AA, n_points=200):
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

        nu_min = (c/lambda_max)
        nu_max = (c/lambda_min)

        nu_grid = np.linspace(nu_min, nu_max, n_points)
        lamb_grid = (c/nu_grid) * u.cm
        Fnu_list = []
        # pre-compute the continuum flux (Planck function)
        B_lambda_grid = self.continuum(lamb_grid)#.cgs#.value

        B_nu_grid = convert_flux(lamb_grid, B_lambda_grid, out_flux_unit='erg / (Hz s cm2 sr)')#.cgs#.value
        # I will have to convert to cgs units for consistency and then strip the units before calculating spectrum
        Bnu_cgs_unit = B_nu_grid.unit
        #print("Units of B_nu", B_nu_grid.unit)
        #print("Unit of B_\lambda", B_lambda_grid.unit)
        #exit()
        B_nu_grid = B_nu_grid.value

        # In a lot of the spectrum, there may not be any opacity to cause absorption/emission
        # in those parts, just return the continuum. This mask is used to decide that.
        line_masks = self.get_line_mask(nu_grid)

        p_grid = np.linspace(0, self.r_max, 200)
        flat_continuum_grid = np.ones_like(B_nu_grid)
        params = [
            (nu, nu/self.rest_frequencies, line_masks[i], flat_continuum_grid[i], p_grid)
            for i, nu in enumerate(nu_grid)
        ]
    

        t_i = time.time()
        with ProcessingPool(num_cores) as pool:
            Fnu_list = pool.map(lambda args: self.calc_flux_at_nu(*args), params)
        
        print(f"Time taken: {(time.time() - t_i):.3f} seconds for full spectrum calculation")
        Fnu_list = np.array(Fnu_list) * B_nu_grid
        F_lambda = convert_flux(nu_grid * u.Hz, Fnu_list * Bnu_cgs_unit * u.sr, out_flux_unit='1e20 erg / (s AA cm2)')# * u.sr
        wavelength_grid = (lamb_grid).to("AA").value
        return wavelength_grid, F_lambda.value/(np.pi*self.r_min**2)


if __name__ == "__main__":
    # some sanity checking plots
    print("Nothing crashed! Can we celebrate that?")

import multiprocessing as mp
import time
from dataclasses import dataclass
from functools import partial

import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c, h, k_B
from astropy.modeling.physical_models import BlackBody
from astropy.units import Quantity
from scipy.integrate import quad
from synphot.units import convert_flux

"""
Does the PCygni line formation while taking into account the level populations,
and optical depths not just at the photosphere but at each point on a grid, taken in the velocity space
of the ejecta profile

Author: Aayush
(so you know whom to blame for mistakes)
"""

num_cores = mp.cpu_count()
# There isn't any point in it using GPU when there aren't enough points to calculate.
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_num_cpu_devices', num_cores)
# to use 64 bit floats?
#jax.config.update('jax_enable_x64', True)
#print("Number of CPU cores available", jax.device_count())

# I don't like cgs units, but we're doing astronomy so I have to live with it
c = c.cgs.value
h = h.cgs.value
k_B = k_B.cgs.value


@jax.jit
def fast_interpolate(x_new: float, v_grid: np.array, y_grid: np.array):
    """
    A fast interpolation function to be used in the numba-compiled code
    """
    return jnp.interp(x_new, v_grid, y_grid)

@jax.jit
def W(r, r_min):
    """
    The geometric dilution factor.
    """
    return (1 - jnp.sqrt(1 - (r_min/r)**2)) / 2

@jax.jit
def source_function(nu, v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0, continuum, photosphere, mode='nLTE'):
    """
    The source function, defined as the ratio of the emissivity to the absorption coefficient
    has the following form.
    """
    if mode == 'LTE':
        #T = photosphere.continuum.temperature.value
        return continuum#(2 * h * nu_0**3 / c**2) / (jnp.exp(h * nu_0 / (k_B * T)) - 1)
    n_u = jnp.interp(v / c, v_grid, n_u_grid)
    n_l = jnp.interp(v / c, v_grid, n_l_grid)
    return (2 * h * nu_0**3 / c**2) / (g_u * n_l / (g_l * n_u) - 1)

@jax.jit
def S(nu, p, z, r, r_min, r_max, v, sob_esc_prob, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0, continuum, photosphere, mode='level-pop'):

    outside_photosphere = (r < r_min) | (r > r_max)
    occulted_region = (z < 0) & (p < r_min)
    if mode == 'level-pop':
        # the level populations are given from the NLTE calculation
        return jnp.where(outside_photosphere | occulted_region,
                        1e-10,
                        sob_esc_prob*source_function(nu, v, g_u, g_l, n_u_grid, n_l_grid, v_grid, nu_0, continuum, photosphere)#/np.pi
                        )
    else:
        return jnp.where(outside_photosphere | occulted_region,
                    1e-10,
                    sob_esc_prob * W(r, r_min)*continuum
                    )


@partial(jax.tree_util.register_dataclass,
         data_fields=['tau_grid_eq', 'tau_grid_polar', 'escape_prob_eq', 'escape_prob_polar',
                      'velocity_grid', 'g_upper', 'g_lower', 'n_upper', 'n_lower'],
         meta_fields=['wavelength'])
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
        self.nu_0 = (self.wavelength * u.cm).to("Hz", equivalencies=u.spectral())#.value
        self.wavelength = self.wavelength#.value
        # compile an interpolation grid for tau, n_upper and n_lower
        # later, one can sample from these while integrating.
        self.n_u = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_upper)
        self.n_l = lambda v: fast_interpolate(v/c, self.velocity_grid, self.n_lower)
        self._tau_interp_eq = lambda v: fast_interpolate(v/c, self.velocity_grid, self.tau_grid_eq)
        self._tau_interp_polar = lambda v: fast_interpolate(v/c, self.velocity_grid, self.tau_grid_polar)
        self.sob_esc_prob_eq = lambda v: fast_interpolate(v/c, self.velocity_grid, self.escape_prob_eq)
        self.sob_esc_prob_polar = lambda v: fast_interpolate(v/c, self.velocity_grid, self.escape_prob_polar)

 

@jax.jit
def calc_r(p: float, z: float):
    return jnp.sqrt(p**2 + z**2)

@jax.jit
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
    B = jnp.where(jnp.isnan(B), 0., B)
    z = c * t * A * (1 - B)
    return z

@jax.jit
def is_polar_ejecta(p, z, polar_opening_angle, observer_inclination_angle):
    """
    For implementing the two-component model, check if the point (p, z)
    is within the "polar" ejecta region or not.
    Also takes into account an observer's inclination angle
    """
    beta = observer_inclination_angle
    p_new = p * jnp.cos(beta) + z * jnp.sin(beta)
    z_new = -p * jnp.sin(beta) + z * jnp.cos(beta)
    return jnp.where((jnp.abs((jnp.arctan(p_new/z_new))) < polar_opening_angle/2), 1, 0)

@jax.jit
def I(p, continuum, r_min):
    """
    The incident specific intensity beam
    """
    return jnp.where(p < r_min, continuum, 0.)
    
@partial(jax.tree_util.register_dataclass, 
         data_fields=['polar_opening_angle', 'observer_angle', 'v_phot', 'v_max', 't_d'],
         meta_fields=['continuum', 'line_list'])
@dataclass
class Photosphere:
    """
    Note: all the quantities have to be converted to cgs units before passing them here.
    """
    v_phot: float
    v_max: float
    t_d: float # time since explosion, needs to be in seconds.
    continuum: BlackBody# Doesn't need to be blackbody tbh. Note that however, for modelling nebular phase
                        # what instead we need to worry about are more general forms of the source function/emissivity
                        # including the non-scattering contribution G(r), etc.
    line_list: list[LineTransition]
    polar_opening_angle: float = np.pi/4
    observer_angle: float = np.pi/2

    def __post_init__(self):
        self.r_max = self.v_max * self.t_d#).to("cm").value
        self.r_min = self.v_phot * self.t_d#).to("cm").value
        self.line_wavelengths = np.array([line.wavelength for line in self.line_list])# * u.cm
        self.rest_frequencies = (self.line_wavelengths * u.cm).to("Hz", equivalencies=u.spectral()).value
        #self.v_phot = self.v_phot.to("cm/s").value
        #self.v_max = self.v_max.to("cm/s").value
        #self.t_d = self.t_d.to("s").value
        #print("Initializing Photosphere for lines", self.line_wavelengths*1e7)
    

    def visualize_polar_region(self):
        fig, ax = plt.subplots()

        v_phot_circle = plt.Circle((0, 0), self.v_phot/c, ec='w',ls='--', fill=False, alpha=1)
        v_max_circle = plt.Circle((0, 0), self.v_max/c, ec='w', ls=':', fill=False, alpha=1)

        p_list = np.linspace(-self.v_max/c, self.v_max/c, 500)
        z_list = np.linspace(-self.v_max/c, self.v_max/c, 500)#/(c*self.t_d)

        Z, P = np.meshgrid(z_list, p_list)#/(c*self.t_d)
        R = calc_r(P, Z)

        X = P
        Y = np.zeros_like(X) # a meridonial slice along \phi=0

        inside_polar = self.is_polar(X, Y, Z, R)#is_polar_ejecta(P, Z, self.polar_opening_angle, self.observer_angle)
        inside_polar &= (R) <= self.v_max/c
        print("Polar‐cone fraction:",
                    inside_polar.mean(),  # fraction of pixels in the
                    "→ solid‐angle fraction ≈", 1 - np.cos(self.polar_opening_angle/2))
        pos = ax.imshow(inside_polar, extent=(-self.v_max/c, self.v_max/c, -self.v_max/c, self.v_max/c), 
                        cmap='coolwarm_r', origin='lower')
        fig.colorbar(pos, ax=ax,label='Polar Ejecta')
        
        ax.add_artist(v_phot_circle)
        ax.add_artist(v_max_circle)

        p_0 = np.zeros_like(p_list)# * self.r_max#/(c*self.t_d)
        ax.plot(z_list, p_0, c='w', lw=1, ls='--')

        ax.text(0.7*self.v_phot/c, 0.02, s=r'$\to$', va='center', ha='center', color='w')
        
        ax.set_xlabel("z (c)")
        ax.set_ylabel("p (c)")
        plt.tight_layout()
        ax.set_aspect('equal')
        plt.savefig(f"polar_ejecta_{self.observer_angle:.2f}.png", dpi=300)
        plt.show()

    @jax.jit
    def tau(self, line, p, z, r, v, is_polar):
        # including the relativistic correction for tau from Hutsemekers & Surdej (1990)
            mu = z / r
            beta = v/c
            corr = abs( (1-mu*beta)**2/( (1-beta)*(mu*(mu-beta)+(1-mu**2)*(1-beta**2))) )

            # decide between polar and equatorial ejecta
            tau_ = jnp.where(is_polar,#is_polar_ejecta(p,z, self.polar_opening_angle, self.observer_angle),
                        corr*line._tau_interp_polar((v/c)),
                        corr*line._tau_interp_eq((v/c))
                        )

            outside_photosphere = (r <= self.r_min) | (r > self.r_max)
            occulted_region = (z < 0) & (p < self.r_min)
            return jnp.where( 
                            outside_photosphere | occulted_region,
                            1e-15,
                            tau_,
                        )

    @jax.jit
    def is_polar(self, x, y, z, r):
        beta = self.observer_angle
        # tilt, for the polar ejecta cone by the observer angle
        x_rot = x * jnp.cos(beta) + z*jnp.sin(beta)
        y_rot = y
        z_rot = -x * jnp.sin(beta) + z * jnp.cos(beta)
        inside_polar = jnp.abs(jnp.arcsin(jnp.sqrt(x_rot**2 + y_rot**2)/r)) <= self.polar_opening_angle/2
                        #jnp.abs(z_rot / r) >= jnp.cos(self.polar_opening_angle/2)
        return inside_polar

    @jax.jit
    def I_emit(self, nu: float, p: float, phi: float, delta_arr: jnp.array, continuum: float):
        z_arr = calc_z(p, self.t_d, delta_arr)
        r_arr  = calc_r(p, z_arr)
        v_arr = r_arr/self.t_d

        # project to sky-plane coordinates
        x = p * jnp.cos(phi) 
        y = p * jnp.sin(phi)
        # check if it falls within the "polar" cones or not
        inside_polar = self.is_polar(x, y, z_arr, r_arr)

        # For the scattering part, the line summation has to be done in the minus \hat{n} direction (i.e. going inwards
        # from observer's line of sight to the center of the explosion, instead of inside to out); see eqn (22) of Jeffrey & Branch (1990)
        # to do the ordering, you have to know where the Sobolev resonance plane lies for each particular line;
        # and that's just 'z' in the impact geometry.
        order = jnp.argsort(z_arr)[::-1]
        # a helper function, because the S function is a bit long
        # TODO: the sobolev escape probability is to be be allowed to be different in the poles and the equator
        S_i = lambda line, p, z, r, v: S(nu, p, z, r, self.r_min, self.r_max, v, line.sob_esc_prob_eq(v/c), line.g_upper, line.g_lower, line.n_upper, line.n_lower, line.velocity_grid, line.nu_0, continuum, self)  
        
        tau_i = jnp.zeros_like(z_arr)
        _Si = jnp.zeros_like(z_arr)
        
        for i, line in enumerate(self.line_list):
            tau_i = tau_i.at[i].set(self.tau(line, p, z_arr[i], r_arr[i], v_arr[i], inside_polar[i]))
            _Si = _Si.at[i].set(S_i(line, p, z_arr[i], r_arr[i], v_arr[i]))
        
        tau_i = jnp.array(tau_i)[order]
        _Si = jnp.array(_Si)[order]
        
        I_abs = I(p, continuum, self.r_min)*jnp.exp(-jnp.sum(tau_i))
        I_scat = jnp.sum(_Si * (1 - jnp.exp(-tau_i)) * jnp.exp(-jnp.cumsum(tau_i) + tau_i))

        return (I_abs + I_scat)


    '''@jax.jit
    def calc_flux_at_nu(self, nu: float, delta_arr: np.array, line_mask: np.array, p_grid: np.array, B_nu: float):
        F_continuum = jnp.pi * self.r_min**2 * B_nu
        def line_flux():
            I_vals = jax.vmap(lambda p: p * self.I_emit(p, delta_arr, B_nu))(p_grid)
            return 2*np.pi* jnp.trapezoid(I_vals, p_grid)
        return jnp.where(line_mask, F_continuum, line_flux())'''
    @jax.jit
    def calc_flux_at_nu(self, nu: float, delta_arr: jnp.ndarray, line_mask: bool, p_grid: jnp.ndarray, B_nu: float):
        # immediate continuum case (because there are no lines to calculate)
        F_cont = jnp.pi * self.r_min**2 * B_nu
        def line_flux():
            Nphi = 200
            phi = jnp.linspace(0.0, 2*jnp.pi, Nphi, endpoint=False)
            I_p_phi = jax.vmap(lambda p: jax.vmap(
                                    lambda phi: self.I_emit(nu, p, phi, delta_arr, B_nu)
                                )(phi), in_axes=(0,)
                            )(p_grid)
            J_p = jnp.trapezoid(I_p_phi, phi, axis=1)
            return jnp.trapezoid(p_grid * J_p, p_grid)
        return jnp.where(line_mask, F_cont, line_flux())
    
    def calc_spectral_flux(self, nu_grid, line_masks, B_nu_grid, p_grid):
        delta_arr = nu_grid[:, None] / jnp.array(self.rest_frequencies)[None,:]
        calc_one = lambda nu, delta, line_mask, B: self.calc_flux_at_nu(nu, delta, line_mask,  p_grid, B)
        #params = [
        #    (nu, nu/self.rest_frequencies, line_masks[i], B_nu_grid[i], p_grid)
        #    for i, nu in enumerate(nu_grid)
        #]
        #with ProcessingPool(num_cores) as pool:
        #    Fnu_list = pool.map(lambda args: calc_one(args), params)
        calc_Fnu_all = jax.vmap(
            lambda nu, delta, mask, Bnu: 
            self.calc_flux_at_nu(nu, delta, mask, p_grid, Bnu),
            in_axes=(0, 0, 0, 0)
        )
        Fnu_list = calc_Fnu_all(nu_grid, delta_arr, line_masks, B_nu_grid)       #np.array([self.calc_Fnu(*args) for args in params])
        return Fnu_list


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

    def calc_spectrum(self, start_wav=2_350*u.AA, end_wav=24_500*u.AA, n_points=200):
        """
        I was thinking, if I pass v_phot and v_max here that can be used for a fitting routine
        it can be used to update v_phot and v_max set in the Photosphere object
        """

        # It may happen that the user specified wavelength range actually does not include the bluest or reddest part of the line
        lambda_min =(1-self.v_phot/c)*np.min(self.line_wavelengths) * 0.95
        lambda_max = (1+self.v_max/c)*np.max(self.line_wavelengths) * 1.05
        # in that case, just do the calculation in regions outside it.
        lambda_min = np.min([start_wav.to("cm").value, lambda_min])
        lambda_max = np.max([end_wav.to("cm").value, lambda_max])

        nu_min = (c/lambda_max)
        nu_max = (c/lambda_min)

        nu_grid = np.linspace(nu_min, nu_max, n_points)
        lamb_grid = (c/nu_grid) * u.cm
        Fnu_list = []

        # pre-compute the continuum flux (passed as a BlackBody object)
        B_nu_grid = self.continuum(nu_grid * u.Hz)

        # I will have to convert to cgs units for consistency and then strip the units before calculating spectrum 
        Bnu_cgs_unit = B_nu_grid.unit
        B_nu_grid = B_nu_grid.value

        # In a lot of the spectrum, there may not be any opacity to cause absorption/emission
        # in those parts, just return the continuum. This mask is used to decide that.
        line_masks = self.get_line_mask(nu_grid)

        p_grid = np.linspace(0, self.r_max, 200)
        #print("Starting flux integration!")
        t_i = time.time()
        Fnu_list = self.calc_spectral_flux(nu_grid, line_masks, B_nu_grid, p_grid)

        #t_i = time.time()
        Fnu_list = np.array(Fnu_list)
        F_lambda = convert_flux(nu_grid * u.Hz, Fnu_list * Bnu_cgs_unit * u.sr, out_flux_unit='erg /(s AA cm2)')
        wavelength_grid = (lamb_grid).to("AA").value
        print(f"Time taken: {(time.time() - t_i):.3f} seconds for full spectrum calculation")
        #print("Took", t_i - time.time(), "to prepare F_lambda")
        #print("Returning now!")
        return wavelength_grid, F_lambda.value/(np.pi*self.r_min**2)


if __name__ == "__main__":
    # some sanity checking plots
    print("Nothing crashed! Can we celebrate that?")

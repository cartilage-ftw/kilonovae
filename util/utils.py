import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.constants import c, h, k_B
from astropy.modeling.physical_models import BlackBody
from astropy.modeling.models import custom_model
from astropy.modeling import Parameter
from lmfit import Model

WIENS_CONSTANT = 2898 * u.Unit('um K')
wiens_law_temperature = lambda wavelength: (WIENS_CONSTANT/wavelength).to('K')


def planck_function(wavelength_grid, T, amplitude, z=0.):
    """
    Note that this returns F_\lambda, not F_\nu
    """
    if not isinstance(wavelength_grid, u.Quantity):
        wavelength_grid = wavelength_grid * u.AA
    if not isinstance(T, u.Quantity):
        T = T * u.K
    B = BlackBody(temperature = T, scale=amplitude*u.Unit("1e20 * erg/(s sr cm2 AA)"))
    wav = wavelength_grid*(1+z)
    wav = np.maximum(wav.value, 1E-10)*u.AA # just to prevent negative values in case a fitter tries to use them
    return np.pi * B(wav) * u.sr# * 4*np.pi*u.sr

def fit_blackbody(wavelength, flux, masked_regions, z=0., requested_T=None):
    """
    Please make sure the wavelength is in Angstroms
    (even if it's just a scalar and not an astropy.Quantity object)
    and the flux has the correct units of 10^-20 * 'erg s-1 cm-2 AA-1'.
      If not, convert them beforehand.
      The 10^-20 is introduced to keep the flux values of order ~1 instead of 10^-20
      (for numerical safety reasons, while fitting)
    """
    blackbody_model = Model(planck_function)

    # there are some nan values in wavelength and flux
    nan_mask = ~np.isnan(flux) & ~np.isnan(wavelength) 
    flux = flux[nan_mask] # keep y-axis values > O(1) for numerical reasons
    wavelength = wavelength[nan_mask]

    print("typical values of flux", np.max(flux))
    # make initial guess of temperature
    peak_wavelength = wavelength[np.argmax(flux)]
    if not isinstance(peak_wavelength, u.Quantity):
        peak_wavelength *= u.AA
    T_guess = wiens_law_temperature(peak_wavelength)

    amp_guess = np.max(flux)/planck_function(peak_wavelength, T_guess.value, 1).value
    print("Wien temperature guess was", T_guess)
    pars = blackbody_model.make_params(z=z,
                T=dict(value=T_guess.value, min=2000., max=15_000.),
                amplitude=dict(value=amp_guess, min=amp_guess*1E-3, max=amp_guess*1E3))

    pars['z'].set(vary=False) # the spectra are dereddened already
    if requested_T is not None:
        pars['T'].set(value=requested_T, vary=False)

    mask = np.ones_like(wavelength, dtype=bool)
    # mask out regions to exclude while fitting
    for left, right in masked_regions:
        mask &= ~((wavelength > left) & (wavelength < right))

    return blackbody_model.fit(flux[mask], params=pars, wavelength_grid=wavelength[mask],
                                                method='differential_evolution')


def calc_luminosity_distance(amplitude: float, v_phot: u.Quantity, t_d: u.Quantity):
    """
    Takes the fit from `fit_blackbody()` and given a time since explosion,
    calculates the 
    """
    R_ph = v_phot * t_d
    D_L = R_ph/np.sqrt(amplitude * 1e-20)
    return D_L.to("Mpc")

scientific_notation = lambda num: r"{:.2f} \times 10^{}".format(num / 10**int(np.log10(num)), int(np.log10(num)))

def display_rate_timescale(rate_matrix, states, process_name, environment, mode='collapse_srI'):
    all_state_names = states.all_names
    if mode == 'collapse_srII':
        num_ion_states = len(states.all_names) - len(states.names) + 1
        t_matrix = np.zeros((num_ion_states, num_ion_states))
        recip_mat = np.reciprocal(rate_matrix)
        k_ = len(states.names)
        for (i,j) in np.ndindex(rate_matrix.shape):
            if i >= k_ and j >= k_:
                t_matrix[i-k_,j-k_] = recip_mat[i,j]
        fig, ax = plt.subplots(figsize=(6,6))
        mat = ax.matshow(t_matrix, cmap='plasma_r')
        plt.show()
    else: 
        t_matrix = np.where(rate_matrix==0., np.nan, rate_matrix)#np.reciprocal(rate_matrix)
        print(np.reciprocal(rate_matrix).tolist())
        fig, ax = plt.subplots(figsize=(6,6))
        mat = ax.matshow(t_matrix, cmap='plasma_r')
        ax.set_xticks(np.arange(t_matrix.shape[0]), all_state_names, rotation='vertical')
        ax.set_yticks(np.arange(t_matrix.shape[0]), all_state_names)

        ax.set_xticks(np.arange(-0.5, t_matrix.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, t_matrix.shape[0], 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

        ax.set_xlabel(f"$t_d={environment.t_d}$ days, $n_e={scientific_notation(environment.n_e)}$" +" cm$^{-3}$")
        plt.colorbar(mat, ax=ax, label=r'$\tau$ [s]')
        plt.title(f"{process_name} Rate")

        plt.tight_layout()

        plt.savefig(f"t_{process_name.lower()}.png")
    #plt.show()

telluric_cutouts = np.array([
		[3000, 4000], # that's an absorption region
		#[5330, 5740], # I don't know why this is cut out from t=1.43d
		[6790, 6970], # NOTE: I chose this by visually looking at the spectra
		#[7490, 7650], # also same; NOTE: remember that telluric subtraction can be wrong
		#[8850, 9700],
        [7000, 9500],
        #[7000,9500],
		[10950, 11600],
		[12400, 12600],
		[13100, 14950], # was 14360
		[17550, 20050] # was 19000
	])

def load_roederer_pattern():
    """
    Returns the r-process template (abudance pattern of the star HD 222925)
    from Roderer et al. 2022 https://arxiv.org/abs/2205.03426
    """
    z_abun_dict = {}
    return z_abun_dict



if __name__ == "__main__":

    load_roederer_pattern()
    
    MUSE_SPEC_PATH = './spectra/MUSE/MUSEspec.dat'

    spectra_dir = './Spectral Series of AT2017gfo/1.43-9.4 - X-shooter/dereddened+deredshifted_spectra/'
    epoch_1 = 'AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase+1.43d_deredz.dat'

    xshooter_spec_1 = np.loadtxt(spectra_dir + epoch_1)
    #xshooter_spec_1[:,1] *= 1E20
    muse_spec = np.loadtxt(MUSE_SPEC_PATH)
    muse_spec[:,1] *= 1E-20 # for consistent units with Xshooter
    muse_spec[:,2] *= 1E-20
    fit_xs = fit_blackbody(xshooter_spec_1[:,0], xshooter_spec_1[:,1], masked_regions=telluric_cutouts)
    fit_muse = fit_blackbody(muse_spec[:,0], muse_spec[:,1], masked_regions=telluric_cutouts)

    print('Result of fit:', fit_xs.fit_report())
    print("MUSE fit:", fit_muse.fit_report())
    fig, ax = plt.subplots()
    wavelength_grid = np.linspace(3500, 22000, 10000)
    ax.plot(xshooter_spec_1[:,0], xshooter_spec_1[:,1])
    ax.plot(muse_spec[:,0], muse_spec[:,1])
    ax.plot(wavelength_grid, fit_xs.eval(wavelength_grid=wavelength_grid), label='blackbody')
    ax.plot(wavelength_grid, fit_muse.eval(wavelength_grid=wavelength_grid), label='MUSE')
    # hide telluric regions
    flux_min_grid = -0.4E-16 * np.ones(100)
    flux_max_grid = 2.5E-16 * np.ones(100)
    for (left, right) in telluric_cutouts:
        horizontal_grid = np.linspace(left, right, 100)
        ax.fill_between(horizontal_grid, flux_max_grid, flux_min_grid, zorder=1, fc='lightgray')

    ax.legend()
    plt.show()

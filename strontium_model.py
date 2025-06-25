import multiprocessing as mp
import os
import time
from collections.abc import Iterable
from fractions import Fraction
from functools import cache, partial

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from astropy.constants import c, h, k_B, m_e
from astropy.modeling import fitting
from astropy.modeling.physical_models import BlackBody
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D
from astropy.units import Quantity
from synphot.units import convert_flux
from pathos.multiprocessing import ProcessingPool
from scipy.integrate import quad
from scipy.interpolate import interp1d

import misc
import NLTE.collision_rates
import NLTE.NLTE_model
import util.atomic_utils as atomic_utils
import util.utils as utils
from line_formation_jax import LineTransition, Photosphere
from NLTE.NLTE_model import (BrokenPowerLawProfile, CollisionProcess, Environment,
                             HotElectronIonizationProcess, NLTESolver,
                             PhotoionizationProcess, RadiativeProcess,
                             RecombinationProcess, States)
from pcygni_5_Spec_Rel import PcygniCalculator

num_cores = mp.cpu_count()

# truncate all data above this level (for now)
MAX_ENERGY_LEVEL = 25_000.0 # cm-1, corresponds to 400 nm from ground state.
# other than Sr II
ionization_stages_names = ['Sr I', 'Sr III', 'Sr IV', 'Sr V']

# strontium mass fraction (not just Sr II, all stages)
#mass_fraction = 0.0002 # for initialization, will be fitted later

spectra_root= './Spectral Series of AT2017gfo/'
xshooter_dir = spectra_root + '1.43-9.4 - X-shooter/dereddened+deredshifted_spectra/'

file_idx = lambda day: [str(day) in name for name in os.listdir(xshooter_dir)].index(1)
file_names = os.listdir(xshooter_dir)
# call xshooter_data(1.43) or xshooter_data(1.4) to get the spectrum of t=1.43 days
xshooter_data = lambda day: np.loadtxt(xshooter_dir + file_names[file_idx(day)])
# NOTE: This assumes there's only one file with e.g. '+1.43d' in its filename in the dir

# and same shit with SOAR spectra
soar_dir = spectra_root + '1.47-10.5 days - Gemini + SOAR/'
get_soar_file = lambda day: [str(day) in name for name in os.listdir(soar_dir)].index(1)
#soar_spec = lambda day: np.loadtxt(soar_dir + get_soar_file(day), )

# and ANU spectrum of 0.92 days
anu_spec = np.loadtxt(spectra_root + "0.92 days - ANU/anufit_spectrum_dered.dat", skiprows=1)

# utility functions used later, to access properties of e.g. RadiativeProcess or RecombinationProcess
get_process = lambda solver, process: [p for p in solver.processes if isinstance(p, process)][0]
get_rate_matrix = lambda solver, process: get_process(solver, process).get_transition_rate_matrix()

# just used the same as albert; found elsewhere in the code
telluric_cutouts = np.array([
    #[3000, 4500], # that's an absorption region
    #[5330, 5740], # I don't know why this is cut out from t=1.43d
    [6790, 6970], # NOTE: I chose this by visually looking at the spectra
    #[7070, 7300], # same as above
    [7490, 7650], # also same; NOTE: remember that telluric subtraction can be wrong
    [9150, 9700],
    [10950, 11600],
# I have been a bit generous in the choice, although the subtraction distorts the continuum
# in a broader range than this

    # [9940, 10300], # why was this masked?
    [12400, 12700],
    [13100, 14950], 
    [17550, 20600]
])

all_masked_regions = np.append(telluric_cutouts,[
                            [3200, 4000],
                            (7000, 10500)
                        ], axis=0)

def get_names(levels_df: pd.DataFrame) -> list:
    configs = levels_df['Configuration'].apply(lambda s: s.split('.')[1])
    return (configs + ' ' +  levels_df['Term'] + ' ' + levels_df['J']).tolist()



# TODO: I had defined a method in atomic_utils.py that was similar but used pd.DataFrame; I should remove redundancy
def get_LTE_pops(states: States, electron_temperature: float) -> np.array:
    if not isinstance(electron_temperature, Quantity):
        electron_temperature *= u.K
    beta = 1/(k_B * electron_temperature)
    Z = atomic_utils.partition_func(load_strontium_levels(), electron_temperature)
    return states.multiplicities * np.exp(-beta*states.energies) / Z

# TODO: move to atomic_utils.py
def compute_LTE_ionizations(n_e: Quantity, T: Quantity, ionization_energies: np.array, species: list) -> np.array:
    if not isinstance(T, Quantity):
        T *= u.K
    if not isinstance(n_e, Quantity):
        n_e /= u.cm**3
    part_funcs = [atomic_utils.partition_func(load_strontium_levels(ion_stage), T) for ion_stage in species]
    saha_factors = np.ones_like(part_funcs)
    alpha = (1/h**3)*(2*np.pi*m_e*k_B * T)**(3/2)
    for i in range(len(species)-1):
        saha_factors[i+1] = 2*(part_funcs[i+1]/part_funcs[i])*(alpha/n_e) \
                                *np.exp(-ionization_energies[i]/(k_B*T))
    # the ionization fractions are dependent on successively lower species
    saha_factors = np.cumprod(saha_factors)
    # normalize
    saha_factors /= saha_factors.sum()
    return saha_factors


def pcygni_interp(wav_to_interp_at, v_max, v_phot, tau, resonance_wav, vref=0.22, ve=0.2, t_0=(1.43*u.day).to('s')):
    wav_grid, pcygni_profile = PcygniCalculator(t=t_0, vmax=v_max * c,
                                 vphot=v_phot * c, tauref=tau, vref=vref *c,
                                 ve=ve * c, lam0=resonance_wav).calc_profile_Flam(npoints=100, mode='both')
    # the PCygni calculator evaluated the profile at certain points it decided
    # we need to interpolate between these to obtain values we want to plot the profile at
    interpolator = interp1d(wav_grid, pcygni_profile, bounds_error=False, fill_value=1)
    return interpolator(wav_to_interp_at)


def get_line_transition_data(tau_matrix, beta_matrix, states, level_occupancy):
    """
    Takes the N x N tau matrix, and gives a list of optical depths
        (and corresponding line wavelengths)
    """
    optical_depths = []
    line_wavelengths = []
    g_upper = []
    g_lower = []
    n_upper = []
    n_lower = []
    escape_probs = []
    #print("SHAPE OF THE TAU MATRIX", tau_matrix.shape)
    for i in range(tau_matrix.shape[0]):
        for j in range(tau_matrix.shape[0]):
            if i <= j: continue
            optical_depths.append(tau_matrix[i,j])
            escape_probs.append(beta_matrix[i,j])
            line_wavelengths.append(np.abs(states.energies[i] - states.energies[j]) \
                                        .to('AA',equivalencies=u.spectral()))
            if states.energies[i] > states.energies[j]:
                g_upper.append(states.multiplicities[i])
                g_lower.append(states.multiplicities[j])
                n_upper.append(level_occupancy[i])
                n_lower.append(level_occupancy[j])
            else:
                g_upper.append(states.multiplicities[j])
                g_lower.append(states.multiplicities[i])
                n_upper.append(level_occupancy[j])
                n_lower.append(level_occupancy[i])
    return optical_depths, escape_probs, line_wavelengths, g_upper, g_lower, n_upper, n_lower


def blackbody_with_pcygnis(wavelength_grid, taus, line_wavelengths, planck_continuum, t_0, v_out, v_phot, vref=0.22,
                                    ve=0.2, occul=1., redshift=0., display=False):
    # if a list or np.array is passed instead of one line
    pcygni_profiles = []
    if isinstance(taus, Iterable): 
        for tau, line_res in zip(taus, line_wavelengths):
            line_adjust = pcygni_interp(wavelength_grid, v_out, v_phot, tau, line_res, vref=vref,
                               ve=ve, t_0=t_0)
            # if accounting for time delay/reverberation
            #line_adjust[line_adjust>1] = (line_adjust[line_adjust>1]-1)*occul + 1
            pcygni_profiles.append(line_adjust)                                                                               
    pcygni_profiles = np.array(pcygni_profiles) # these are normalized to 1. 
    product_profiles = np.prod(pcygni_profiles, axis=0)

    # i wanted to visualize what it was doing
    if display == True:
        fig2, ax2 = plt.subplots()
        for ii, profile in enumerate(pcygni_profiles):
            ax2.plot(wavelength_grid, profile, label=f'{line_wavelengths[ii].value:.2f}' + r'$\mathrm{\AA}$')
        ax2.plot(wavelength_grid, product_profiles, c='k', label='ALL!')
        handles, labels = ax2.get_legend_handles_labels()
        plt.text(x=18000, y=0.9*np.max(product_profiles), s=f"$t={t_0.to('day')}$ days")
        h, l = zip(*sorted(zip(handles, labels),
                    # sort the legend display
                    key = lambda x: x[1]))
        ax2.legend(h, l, loc='lower right')
        plt.show()
    # rescale the flux to match the blackbody continuum and apply the combined PCygni line profiles
    return product_profiles*planck_continuum.eval(wavelength_grid=wavelength_grid)


sr_filepath_dict = {"Sr I": "./atomic_data/SrI_levels_NIST.txt",
     "Sr II": "./atomic_data/SrII_levels_NIST.csv",
     "Sr III": "./atomic_data/SrIII_levels_NIST.txt",
     "Sr IV": "./atomic_data/SrIV_levels_NIST.txt",
     "Sr V": "./atomic_data/SrV_levels_NIST.txt"}

@cache
def load_strontium_levels(ion_stage='Sr II', MAX_ENERGY_LEVEL=None) -> pd.DataFrame:
    # Energy levels from NIST
    levels = atomic_utils.loadEnergyLevelsNIST(file_path=sr_filepath_dict[ion_stage])
    # there's an odd-parity state of the 4p6.8p J=1/2 whose energy is not listed in NIST. This has been discarded
    # it shouldn't matter as it's around ~75,000 cm-1 anyways

    # the ionization limit is included in this, so remove that and keep it separate	
    ionization_limit = levels[levels['Configuration'].str.contains('Sr')] # Sr III ionization limit
    levels.drop(ionization_limit.index, inplace=True)
    if MAX_ENERGY_LEVEL is not None:
        # drop all energy levels above a certain energy threshold for now
        return levels[levels['Level (cm-1)'] < MAX_ENERGY_LEVEL]
    else:
        return levels

# NOTE: Loading of the line data happens in NLTE.NLTE_model.py, this is just for states
def get_strontium_states():
    levels = load_strontium_levels('Sr II', MAX_ENERGY_LEVEL)
    #state_names = SrII_levels['Configuration'] 
    level_energies = levels['Level (cm-1)'].to_numpy() / u.cm
    states_instance = States(names=get_names(levels),
                    multiplicities=levels['g'].to_numpy(),
                    energies=level_energies.to(u.eV, equivalencies=u.spectral()),
                    # for now, I ignore Sr III and only load Sr II levels
                    ionization_species=ionization_stages_names,
                )
    return states_instance

# NOTE: deprecated. TODO: make sure the max level used in NLTE_model.py and the one defined here are the same.
def load_strontium_line_data():
    # A-values for radiative line transitions
    SrII_lines_NIST_all = atomic_utils.loadRadiativeTransitionsNIST('./atomic_data/SrII_lines_NIST_all.csv',
                                                sep=',')

    SrII_lines_NIST = SrII_lines_NIST_all.dropna(subset=['Aki(s^-1)'])
    # drop lines for all upper levels above max_energy_level
    SrII_lines_NIST = SrII_lines_NIST[SrII_lines_NIST['Ek(cm-1)'].astype(float) < MAX_ENERGY_LEVEL]

#@numba.njit
def calc_spectrum_residuals(observed_spectrum, model_spectrum, model_wav_grids, param_name_index,
                            to_show_residuals=False):
    # to make it more appropriate, one can either interpolate between points in the model spectra
    # or request the spectrum calculations to be done especially at points in the observed spectra.
    
    # evaluate the synthetic spectrum on the same grid as the observed spectrum
    nan_free = ~np.isnan(observed_spectrum[:,1])

    eval_range = [6500, 9800]
    to_compare = np.where( nan_free &
                                (observed_spectrum[:,0] > eval_range[0]) & 
                                (observed_spectrum[:,0] < eval_range[1])
                                 
                            )[0]
    
    model_interp = interp1d(model_wav_grids, model_spectrum)(observed_spectrum[to_compare,0])
    residuals = (observed_spectrum[to_compare,1] - model_interp)#/observed_spectrum[to_compare,1]
    
    if to_show_residuals:
        fig, axes = plt.subplots(2,1, figsize=(6,5), height_ratios=[4,1])
    
        axes[0].plot(observed_spectrum[to_compare,0], model_interp, label='interpolated')
        #axes[0].plot(model_wav_grids, model_spectrum, label='synthetic')
        axes[0].plot(observed_spectrum[to_compare,0], observed_spectrum[to_compare,1], label='observed')
        axes[1].plot(observed_spectrum[to_compare,0], residuals)
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].set_title("Density Profile: " + str(param_names[param_name_index]))
        plt.tight_layout()
        plt.show()

    return np.sum(residuals**2)


def display_best_fit_density_profiles(v_grid, ne_profiles, residuals, density_param_names):
    """
    Display what the different density profiles are like, colored by the residuals
    """
    fig, _ax = plt.subplots()
    alphas = np.array(residuals)
    # rescale between 1e-4 and 1.
    alphas = np.interp(alphas, (alphas.min(), alphas.max()), [1e-4, 1])
    print(alphas)

    colors = mpl.colormaps['Spectral'](np.linspace(0., 1., len(density_param_names)))
    best = np.argmin(residuals)
    for ii, profile in enumerate(ne_profiles):
        _ax.plot(v_grid, profile, alpha=alphas[ii], c='mediumpurple', label=density_param_names[ii])
    _ax.set_yscale("log")
    _ax.legend(ncols=1)
    _ax.set_title("Best: " + str(density_param_names[best]))
    _ax.set_xlabel("velocity [c]")
    _ax.set_ylabel("$n_e$ at $t=1$ day")
    #plt.savefig("different_power_laws.png", dpi=300)

M_ejecta = 0.04
atomic_mass = 88

tau_phots_epochs = []
absorption_region = (7000, 9700)
if __name__ == "__main__":
    #misc.plot_electron_density(t_d=1.43, v_phot=0.235, v_out=0.6)
    SrII_states = get_strontium_states()
    SrII_states.texify_names() # TODO: make this get called automatically post-init in NLTE_model.py

    fig, ax = plt.subplots(figsize=(8,6))

    T_elec_epochs = {1.17: 5200,
                    1.43: 4400, # 4400
                    2.42: 3200, # 3200
                    3.41: 2900, # 2900
                    4.40: 2700} # 2800
    aayush_colors = ['slategray'] + list(mpl.colormaps['Spectral'](np.linspace(0, 1., len(T_elec_epochs)-1)))

    # put the telluric masks
    flux_min_grid = -5.5E-16 * np.ones(100)
    flux_max_grid = 6.5E-16 * np.ones(100)
    for (left, right) in telluric_cutouts:
        horizontal_grid = np.linspace(left, right, 100)
        ax.fill_between(horizontal_grid, flux_max_grid, flux_min_grid, fc='silver', alpha=0.25)
    
    triplets = np.array([10917.8, 10039.4, 10330.14])
    # mark the rest wavelength
    #ax.vlines(triplets, np.min(flux_min_grid), np.max(flux_max_grid),
    #                     ls='--', lw=1., ec='k', alpha=0.8)

    '''for (left, right) in [[6900, 11900]]:
        horizontal_grid = np.linspace(left, right, 100)
        f_max_grid = 4e-16*np.ones_like(horizontal_grid)
        f_min_grid = -2e-16*np.ones_like(horizontal_grid)
        ax.fill_between(horizontal_grid, f_max_grid, f_min_grid, fc='silver', alpha=0.25)
        ax.text(9_000, -1e-16,r"1$\mu$m feature", c='dimgray', ha='center', va='center')
        ax.vlines([left, right], ymin=-2e-16, ymax=4e-16, ec='silver', alpha=0.6)'''
    # now plot the spectra, blackbody + pcygni fits
    wavelength_grid = np.linspace(2350, 23000, 10_000) * u.AA

    T_phots = [5200, 4400, 3200, 2900, 2800]

    
    offsets = np.array([+2.8, -0.4, -2., -3.5, -5])*1E-16
    scale = np.array([0.8, 1.5, 2.5, 3., 3.5])
    #v_outs = [0.4, 0.4, 0.35, 0.3, 0.28]#[0.45, 0.425, 0.35, 0.286, 0.25]
    v_outs = [0.42, 0.42, 0.35, 0.28, 0.26]
    v_phots = [0.25, 0.24, 0.21, 0.17, 0.15]
    #v_phots = [0.26, 0.24, 0.22, 0.18, 0.16]#[0.236, 0.19, 0.18, 0.162]#[0.2, 0.13, 0.12, 0.11]
    ve_s = [0.32] * 5 
    v_refs = [0.2] * 5
    mass_fractions = 0.004*np.ones_like(v_phots)##[0.003, 0.0001, 0.0025, 0.1, 0.2]#[0.00045, 0.03, 0.15, 0.4]#[0.00055, 0.0015, 0.0075, 0.007]#[0.00008, 0.00004, 0.00015, 0.0002]

    #misc.plot_ionization_temporal(np.array([4400, 3200, 2900, 2800]), np.array([1.43, 2.42, 3.41, 4.40]),
    #                       np.array(v_phots))
    #tau_wavelength = None

    def _compute_tau_shell_sr(v_line, n_e, n_Sr, epoch, v_phot, T_phot, T_electrons, mass_fraction):
        atomic_mass=88
        #print("epoch", epoch, "v_line", v_line, "n_e", n_e, "n_Sr", n_Sr)
        #print(f"Initializing t={epoch} for line_velocity", v_line, "T_phot", T_phot, "T_electrons", T_electrons,
        #      "mass fraction", mass_fraction, "v_phot", v_phot, 'n_e', n_e, 'n_Sr', n_Sr)
        env = Environment(t_d=epoch, T_phot=T_phot, mass_fraction=mass_fraction,
                          atomic_mass=atomic_mass, n_e=n_e, n_He=n_Sr, photosphere_velocity=v_phot,
                          line_velocity=v_line, T_electrons=T_electrons)

        processes = [
                                RadiativeProcess(SrII_states, env),
                                CollisionProcess(SrII_states, env),
                                HotElectronIonizationProcess(SrII_states, env),
                                RecombinationProcess(SrII_states, env),
                                # PhotoionizationProcess(SrII_states, env)
                            ]
        #print(SrII_states.tex_names)
        #print("Transition rate matrix for collision process:\n", processes[1].get_transition_rate_matrix())
        #utils.display_rate_timescale(processes[1].get_transition_rate_matrix(), SrII_states,
        #                             process_name = processes[1].name, environment=env)

        solver = NLTESolver(env, SrII_states, processes=processes)
        t_arr, pops, tau_mat, beta_mat = NLTE.NLTE_model.solve_NLTE_sob(env, SrII_states, solver, mass_fraction)

        depths, escape_probs, wavelengths, g_up, g_lo, n_up, n_lo = get_line_transition_data(
            tau_mat, beta_mat, SrII_states, pops[:, -1]
        )
        return depths, escape_probs, pops[:, -1], g_up, g_lo, n_up, n_lo, wavelengths
    

    
    def compute_tau_LTE(v_line, n_e, epoch, T_phot, mass_fraction, v_phot, atomic_mass):

        env = Environment(t_d=epoch, T_phot=T_phot, mass_fraction=mass_fraction,
                          atomic_mass=atomic_mass, n_e=n_e, photosphere_velocity=v_phot,
                          line_velocity=v_line, T_electrons=T_phot)
        rad_process = RadiativeProcess(SrII_states, env)
        
        lte_pops = get_LTE_pops(SrII_states, electron_temperature=T_phot)


        ion_states = ['Sr I', 'Sr II', 'Sr III', 'Sr IV', 'Sr V']
        sr_ion_energies = np.array([5.70, 11.03, 42.88, 56.28]) * u.eV

        lte_ionizations = compute_LTE_ionizations(env.n_e, env.T_phot, sr_ion_energies, ion_states)
        srII_frac = lte_ionizations[1]/np.sum(lte_ionizations)
        tau_lte, beta_lte = NLTE.NLTE_model.estimate_LTE_tau(env, SrII_states, lte_pops, srII_frac, rad_process,
                                         mass_fraction)
        depths, escape_probs, wavelengths, g_up, g_lo, n_up, n_lo = get_line_transition_data(
            tau_lte, beta_lte, SrII_states, lte_pops,
        )
        return depths, escape_probs, lte_pops, g_up, g_lo, n_up, n_lo, wavelengths

    tau_v = lambda v, vphot, tauref, ve: tauref*np.exp((vphot-v)/ve)
    #misc.plot_velocity_shells(T_elec_epochs.keys(), v_phots, v_outs, mass_fractions)
    
    anu_spec[:,1] *= 1e-17
    salt_spec = np.loadtxt("./Spectral Series of AT2017gfo/1.17 days - SALT spectra/SALT_new_eg1_flux_calibrated_r+i.txt")
    #ax.plot(anu_spec[:,0], anu_spec[:,1], ls='-', alpha=0.4, label='$0.92$ days')
    #ax.plot(salt_spec[:,0], 0.7*salt_spec[:,1]+2e-16, alpha=0.8, lw=1, c='cornflowerblue', label='$t=1.17$ days')


    v_shells = np.linspace(0.1, 0.5, 100)

    dummy_env = Environment()
    broken_p_law = lambda a1, a2, amplitude=1e7, v_break=0.2, delta=0.01: \
                                dummy_env.normalize_density(
                                    BrokenPowerLawProfile(
                                        exponents=[a1, a2],
                                        v_break_position=v_break,
                                        amplitude=amplitude,
                                        delta=delta)
                                    )
    
    '''SmoothlyBrokenPowerLaw1D(amplitude=amplitude,
                                  x_break = v_break,
                                  alpha_1= a1,
                                  alpha_2 = a2,
                                  delta=delta
                                 )'''
    n_e = lambda ne_0, v_line, p=5: ne_0 * (v_line/0.2)**-p


    param_names = ['$p$=4', '$p$=8']
    #ne_profile_1 = n_e(1.5e8, v_shells, p=5)
    ne_profile_2 = n_e(1.5e8, v_shells, p=4)
    ne_profile_3 = n_e(1.5e8, v_shells, p=8)
    
    ne_profiles = [ne_profile_2, ne_profile_3]
    '''a1_ = [2,3,4,5,7]
    a2_ = [3,4,5,7,10]
    for i,j in np.ndindex((len(a1_), len(a2_))):
        ne_profiles.append(
                broken_p_law(a1_[i], a2_[j])(v_shells)
                )
        param_names.append(f'$p_1$={a1_[i]}, $p_2$={a2_[j]}')
        ne_profiles.append(
            broken_p_law(a1_[i], a2_[j], v_break=0.23, amplitude=1e7)(v_shells)
        )
        param_names.append(f'$p_1$={a1_[i]}, $p_2$= {a2_[j]}, v_break=0.23')
        ne_profiles.append(
            broken_p_law(a1_[i], a2_[j], v_break=0.25, amplitude=1e7)(v_shells)
        )
        param_names.append(f'$p_1$={a1_[i]}, $p_2$= {a2_[j]}, v_break=0.25')'''
    spectrum_residuals = np.zeros(len(ne_profiles))
    #my_ne_profile = broken_p_law(2,7, v_break=0.25, amplitude=1e7)(v_shells)
    my_ne_profile = broken_p_law(2,7, v_break=0.2, amplitude=1e7)(v_shells)

    my_ne_profile2 = broken_p_law(5,5, v_break=0.2, amplitude=1e7)(v_shells)#/10
    #print("n_e", my_ne_profile2.tolist())
    #exit()
    for k, ne_profile in enumerate([my_ne_profile2]):
        for i, (epoch, T_e) in enumerate(T_elec_epochs.items()):
            tau_shells = []
            '''env = Environment(t_d=1, T_phot=4400, mass_fraction=mass_fractions[i],
                            atomic_mass=atomic_mass, photosphere_velocity=0.2,
                            line_velocity=0.2, T_electrons=T_e)

            processes = [
                        RadiativeProcess(SrII_states, env),
                        CollisionProcess(SrII_states, env),
                        HotElectronIonizationProcess(SrII_states, env),
                        RecombinationProcess(SrII_states, env),
                        # PhotoionizationProcess(SrII_states, env)
                    ]
            solver = NLTESolver(env, SrII_states, processes=processes)
            #sr_coll_matr = get_rate_matrix(solver, CollisionProcess)
            #sr_rad_matrix = get_rate_matrix(solver, RadiativeProcess)
            sr_nonthermal_matrix = get_rate_matrix(solver, RecombinationProcess)
            utils.display_rate_timescale(sr_nonthermal_matrix, SrII_states,
                                            'Nonthermal Ionization', env)'''
            #utils.display_rate_timescale(sr_coll_matr, SrII_states.tex_names + ionization_stages_names,
                        #                'Collisional', env)
            #exit()
            #exit()

            mp.context._force_start_method('spawn')
            compute_tau_shell_worker = partial(_compute_tau_shell_sr,
                                            #epoch=epoch,
                                            #T_phot=T_phots[i],
                                                #T_electrons=T_e,
                                                #atomic_mass=88,
                                                #mass_fraction=mass_fractions[i],
                                                #v_phot=v_phots[i]
                                                )
            #v_line, n_e, epoch, v_phot, T_phot, T_electrons, mass_fraction,
            _compute_LTE_tau_worker = partial(compute_tau_LTE,
                                              atomic_mass=88,
                                            #epoch=epoch, T_phot=T_phots[i],
                                            #mass_fraction=mass_fractions[i],
                                            #v_phot=v_phots[i]
                                            )

            with ProcessingPool(num_cores) as pool:
                results = pool.map(lambda params: compute_tau_shell_worker(*params),
                                        [(v_shells[idx],
                                          None,#ne_profile[idx],
                                          None,#mass_fractions[i] * ne_profile[idx],
                                          epoch,
                                          v_phots[i],
                                          T_phots[i],
                                          T_e,
                                          mass_fractions[i])
                                           for idx in range(len(v_shells))])
        
                lte_results = pool.map(lambda params: _compute_LTE_tau_worker(*params),
                                     [(v_shells[v_i],
                                         ne_profile[v_i],
                                         epoch,
                                         T_phots[i],
                                         mass_fractions[i],
                                         v_phots[i],
                                         ) for v_i in range(len(v_shells))])

            print("COUNTING!: ", i, epoch, T_e)
            # 'plum', 'orchid', 'mediumpurple', 'mediumslateblue',
            colors = ['mediumpurple'] + list(mpl.colormaps['coolwarm'](np.linspace(0, 1., len(T_elec_epochs)-1)))
            
            mark_offsets = [-0.9E-16, 0., -2.7E-17, -0.7E-16, -0.9E-16]
            # mark the Sr II triplets
            if True: 
                ax.vlines(triplets * (1-v_phots[i]), 2e-16 + offsets[i] + mark_offsets[i],
                                     3e-16 + offsets[i] + mark_offsets[i],lw=1,
                            ls='--',ec=colors[i], alpha=0.6)
                #a_left, a_right, a_ht, a_vertical = (triplets[0], (-v_phots[i])*triplets[0], offsets[i],0.)
                #ax.arrow(a_left, a_right, a_ht, a_vertical)
                #ax.annotate("", xytext=(0, 0), xy=(triplets[0], offsets[i]), arrowprops=dict(arrowstyle="<-"))
            
            # there's XShooter spectra for 1.4 day onwards
            if epoch >= 1.4:
                spec_ep = xshooter_data(day=epoch)
            # NOTE: Except for 1.17 day spectrum which is from SALT, not XShooter
            elif epoch == 1.17:
                spec_ep = salt_spec
            elif epoch == 0.92:
                spec_ep = anu_spec
            # sets the amplitude, etc. of the fitted blackbody 
            #T_bb = [6368.6,
            #        5379.6, #- 150,
            #        3691.1,
            #        3151.0,
            #        2864.3]
            fitted_cont = utils.fit_blackbody(spec_ep[:,0], spec_ep[:,1], all_masked_regions)
            #continuum_model = utils.BlackBodyFlux()
            #fitter = fitting.LevMarLSQFitter()
            #best_fit_continuum = fitter(continuum_model, spec_ep[:,0], spec_ep[:,1])
            # only thing that needs to be tuned is then the mass_fraction
            #fitted_spectral_line = utils.fit_planck_with_pcygni(spec_ep[:,0], spec_ep[:,1], telluric_cutouts_albert)
            T = fitted_cont.params['T'].value
            T_sigma = fitted_cont.params['T'].stderr
            amplitude = fitted_cont.params['amplitude'].value
            print("Value of amplitude:", amplitude)
            print("Calculated luminosity distance:",
                   utils.calc_luminosity_distance(amplitude/np.pi, v_phots[i] * c, epoch * u.day))
            print(rf"Fitted blackbody temperature: {T:.1f} K")

            t_days = epoch * u.day
            # photospheric radius at which the line formation begins.
            R_ph = (v_phots[i]*c * t_days).to("cm").value
            # the flux is diluted also due to the distance to the observer. This information was contained
            # in the fit. Now, I will later rescale the flux from the line formation code to 
            # match the observer
            flux_scale_amplitude = ((1/R_ph)**2) * amplitude * 1e-20
            # this continuum object will be passed to the line formation code.
            continuum = BlackBody(temperature = T * u.K, scale=1*u.Unit("erg/(s cm2 Hz sr)"))
            
            '''sr_coll_matr = get_rate_matrix(solver, CollisionProcess)
                recombination_matrix = get_rate_matrix(solver, RecombinationProcess)
                nonthermal_matrix = get_rate_matrix(solver, HotElectronIonizationProcess)
                
                utils.display_rate_timescale(recombination_matrix, SrII_states.tex_names + ionization_stages_names,
                                        'Recombination', environment)
                utils.display_rate_timescale(nonthermal_matrix, SrII_states.tex_names + ionization_stages_names,
                                        'Non-thermal Ionization', environment)
            '''
            # plot the spatial profile of tau
            if False:
                misc.plot_tau_shells(tau_shells, tau_wavelength)
                misc.plot_mean_ionization(v_shells, shell_occupancies, epoch)
            print("At t=",epoch, "T_phot", T_phots[i], "T_electrons", T_e, "mass_fraction", mass_fractions[i])
            #misc.print_luminosities(SrII_states, line_luminosities, tau_shells)
            #misc.print_line_things(SrII_states, v_shells, shell_occupancies, tau_shells)
            

            # convert these list objects to a numpy array
            tau_shells, escape_probs, shell_occ, g_upper, g_lower, n_up_grids, n_lower_grid, resonance_wavelengths = zip(*results)
            tau_shells   = np.array(tau_shells)
            shell_occ    = np.array(shell_occ)
            n_upper_grid   = np.array(n_up_grids)
            n_lower_grid   = np.array(n_lower_grid)
            escape_probs = np.array(escape_probs)

            #misc.compare_ionization_profile(v_shells, SrII_states, shell_occ, T, epoch)
            
            tau_lte, escape_prob_lte, shell_occ_lte, g_upper, g_lower, n_upper_lte, n_lower_lte, resonance_wavelengths = zip(*lte_results)
            tau_lte = np.array(tau_lte)
            n_upper_lte = np.array(n_upper_lte)
            n_lower_lte = np.array(n_lower_lte)
            escape_prob_lte = np.array(escape_prob_lte)
            #print(f"Tau at photosphere at epoch {epoch}")
            #print(tau_shells[0,:])
            tau_phots_epochs.append(tau_shells[0,:])
            #print("for lines", resonance_wavelengths[0])
            #exit()
            line_transition_objects = []
            for j, line_wavelength in enumerate(resonance_wavelengths[0]):
                # ignore forbidden lines that are too weak and will not be visible
                if line_wavelength.to("nm").value > 2_000: continue
                #print("Escape probability for", line_wavelength.to("nm").value, "is", escape_probs[:,j])
                line_transition = LineTransition(
                                        (line_wavelength).to("cm").value,
                                        tau_shells[:,j],#np.zeros_like(tau_shells[:,j]),#, # equatorial ejecta #
                                        tau_shells[:,j], # polar ejecta
                                        escape_probs[:,j],
                                        escape_probs[:,j],
                                        v_shells, 
                                        g_upper[0][j], 
                                        g_lower[0][j], 
                                        n_upper_grid[:,j], 
                                        n_lower_grid[:,j]
                                )
                line_transition_objects.append(line_transition)

            lte_transition_objects = [
                            LineTransition(
                                (line_wavelength).to("cm").value,
                                tau_lte[:,j], # equatorial ejecta np.zeros_like(tau_shells[:,j]),#
                                tau_lte[:,j], # polar ejecta
                                escape_prob_lte[:,j],
                                escape_prob_lte[:,j],
                                v_shells, 
                                g_upper[0][j], 
                                g_lower[0][j], 
                                n_upper_lte[:,j], 
                                n_lower_lte[:,j]
                            )
                    for j, line_wavelength in enumerate(resonance_wavelengths[0])
                            if line_wavelength.to('nm').value < 2_000
            ]
            if True:
                # plot the observed spectrum
                nan_mask = ~np.isnan(spec_ep[:, 1])
                if epoch > 1.17:
                    ax.step(spec_ep[nan_mask,0], scale[i]*spec_ep[nan_mask,2]+ offsets[i], ls='-', color='darkgray', alpha=0.4, lw=0.75) #/fitted_cont.eval(wavelength_grid=spec_ep[:,0]) - i 
                ax.step(spec_ep[nan_mask,0], scale[i]*spec_ep[nan_mask,1]+ offsets[i], ls='-', color=colors[i], lw=0.75,
                                alpha=0.8, label=f'$t={epoch}$ days')
                
                observer_angle = 0. #np.pi/4#np.pi/3#np.deg2rad(22)#np.pi/12
                # and then compute and plot syntheti c spectrum
                photosphere = Photosphere(v_phot=(v_phots[i] * c).cgs.value, 
                                        v_max=(v_outs[i] * c).cgs.value,
                                        t_d=(epoch * u.day).cgs.value,
                                        continuum=continuum,
                                        line_list=line_transition_objects,
                                        polar_opening_angle=np.pi/4,
                                        observer_angle=observer_angle)
                
                lte_photosphere = Photosphere(v_phot=(v_phots[i] * c).cgs.value, 
                                        v_max=(v_outs[i] * c).cgs.value,
                                        t_d=(epoch * u.day).cgs.value,
                                        continuum=continuum,
                                        line_list=lte_transition_objects,
                                        polar_opening_angle=np.pi/4,
                                        observer_angle=observer_angle)
                
                full_wav_grid, spectrum_flux = photosphere.calc_spectrum()
                #full_wav_grid, lte_spectrum_flux = lte_photosphere.calc_spectrum()
                #exit()
                spectrum_flux *= flux_scale_amplitude
                ax.plot(full_wav_grid, scale[i]*spectrum_flux + offsets[i],# marker='.',
                                    ls='-.', ms=1, c=aayush_colors[i], lw=1)
                # ,label=f"more physical model t={epoch}"
                
                #ax.plot(full_wav_grid, scale[i]*lte_spectrum_flux + offsets[i], marker='.',
                #                    ls='-', ms=1, c=aayush_colors[i], lw=1.)
                #    print(f"Line {res_lamb:.1f}nm
                #pcygni_line = lambda wav, vphot=v_phots[i], vout=v_outs[i]: blackbody_with_pcygnis(wav, tau_shells[0,:], resonance_wavelengths,
                #                            fitted_cont, t_0=(epoch * u.day).to('s'), v_out=vout, v_phot=vphot, ve=ve_s[i], display=False)
                                    #environment=environment, states=SrII_states, level_occupancy=level_occupancy[:, -2:-1], A_rates=)
                #ax.plot(wavelength_grid, scale[i]*pcygni_line(wavelength_grid) + offsets[i], c='k', ls='-', lw=0.25)
                ax.fill_between(full_wav_grid, scale[i]*spectrum_flux + offsets[i],
                        scale[i]*fitted_cont.eval(wavelength_grid=full_wav_grid) + offsets[i],
                                    fc='#ebc3d4', alpha=0.5) #ebc3d4
                
                spectrum_residuals[k] += calc_spectrum_residuals(np.array(spec_ep), np.array(spectrum_flux), full_wav_grid, k)
                #ax.fill_between(wavelength_grid.value, scale[i]*pcygni_line(wavelength_grid) + offsets[i],
                #        scale[i]*fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
                #                    fc='#ebc3d4', alpha=0.5) #ebc3d4
                # place text stating X(Sr) at an appropriate height
                desired_text_loc = 3_900 # i know, weird choice of number
                if epoch == 1.17:
                    desired_text_loc = 6200
                valid_flux = ~np.isnan(spec_ep[:, 1])  # Mask for valid flux values
                valid_indices = np.where(valid_flux)[0]  # Indices of valid flux
                closest_index = valid_indices[np.argmin(np.abs(spec_ep[valid_indices, 0] - desired_text_loc))]
                loc_flux = spec_ep[closest_index, 1]

                ax.text(x=desired_text_loc, y=offsets[i] + loc_flux -0.1E-16, ha='left',
                            s='$X_{Sr}=' + f'{mass_fractions[i]*100:.3f} \%$', c=colors[i])#s=f"$t={epoch}$ days", c=colors[i])#
                ax.plot(wavelength_grid, scale[i]*fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
                            ls='-', color='dimgray', #label=f"{T:.2f} $\pm$ {T_sigma:.2f}",
                            lw=0.5,)
                
                # overplot Swift-UVOT photometry
                if epoch == 1.17:
                    swift_uvot_phot = np.array([[3431.4303172067657, 3.7387663382677025e-16, 5.4229879381201765e-17, 388.6973735941286],
                    [2574.8106276298963, 2.5913688689382614e-16, 4.9462102531106275e-17, 343.14303172067656],
                    [2224.2402575602873, 5.425938405527584e-17, 2.5380952908126773e-17, 246.58763318455547]])
                    #wav = [2243.15, 2593.87, 3464.83]
                    #y_flux = np.array([5.3e-17, 2.67e-16, 3.51e-16])
                    #y_max = [7.64e-17, 3.18e-16, 4.03e-16]
                    #y_err = np.array(y_max) - np.array(y_flux)
                    #x_range = abs(np.array([1977, 2223, 3041]) - np.array(wav)) 
                    ax.errorbar(swift_uvot_phot[:,0], scale[i]*swift_uvot_phot[:,1] + offsets[i], yerr=scale[i]*swift_uvot_phot[:,2], xerr=swift_uvot_phot[:,3], marker='o',
                                            c='mediumpurple', ls='', ms=3, capsize=2, alpha=0.7)
                
                #ax.set_ylim(top=3.8E-16, bottom=-1.9E-16)
                #ax.set_xlim()
                #ax.set_xscale('log')
                #from matplotlib.ticker import ScalarFormatter
                #ax.set_xticks([4000, 6000, 10000, 14000, 20000])
                #ax.set_yticks([])
                #plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
                #plt.gca().xaxis.set_major_formatter(ScalarFormatter()) 

                #ax.ticklabel_format(axis='x', style='plain')
                plt.savefig(f'fitted_cpygni_{observer_angle:.1f}.png', dpi=300)
                #photosphere.visualize_polar_region()
                # TODO: 
                #display_time_solution(t, level_occupancy, tau_all_timesteps, environment)
        if k == 0:
            pass
            # draw an inset zooming into the spectra.
            '''(left, right, bottom, top) = (6100, 13100, 0.3e-16, 7e-16)
            ax2 = ax.inset_axes([0.5, 0.5, 0.5, 0.5],
                                 xlim=(left, right),
                                   ylim=(bottom, top),
                                     xticklabels=[],
                                     yticklabels=[])
            ax2.grid('off')
            ax2.plot(salt_spec[:,0], scale[0]*salt_spec[:,1] + offsets[0], c='mediumpurple')
            ep1_spec = xshooter_data(1.43)
            ax2.plot(ep1_spec[:,0], scale[1]*ep1_spec[:,1] + offsets[1], c=colors[1])
            left, bottom, width, height = (0.6, 0.6, 0.3, 0.3)
            x2 = fig.add_axes([left, bottom, width, height])
            
            ax2.set_xticks([])
            ax2.set_yticks([])'''
            ax.legend(loc='upper right', title='Time since explosion')
    ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
    ax.set_xlabel("Wavelength [$\mathrm{\AA}$]")
    #ax.set_xscale('log')
    #ax.set_yticks([])
    #ax.set_xlim(left=2400, right=24000)
    #ax.set_ylim(top=8.5E-16)
    
    #ax.set_xscale('log')
    plt.tight_layout()
    #plt.savefig(f'fitted_pcygni.png', dpi=300)
    
    #display_best_fit_density_profiles(v_shells, ne_profiles, spectrum_residuals, param_names)
    """
    _fig, _ax = plt.subplots()
    _ax.plot([1.17, 1.43, 2.42, 3.41, 4.40], np.array(tau_phots_epochs),
             label=[f"{wav.to('nm').value:.1f}" for wav in resonance_wavelengths[0]])
    _ax.legend()
    _ax.set_yscale('log')
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel(r"Sobolev Optical Depth $\tau$")
    """
    plt.show()



param_bounds = {
    'X_Sr': (1, 20),
    'log_ne_grid': ([]),
    'v_phots': (),
    'v_maxs': (),
    #'T_e': (),
}

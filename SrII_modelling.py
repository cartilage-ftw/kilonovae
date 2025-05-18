import os
from collections.abc import Iterable
from fractions import Fraction
from functools import cache

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.constants import c, h, k_B, m_e
from astropy.modeling.physical_models import BlackBody
from astropy.units import Quantity
from lmfit import Model
from scipy.integrate import quad
from scipy.interpolate import interp1d

import main.atomic_utils as atomic_utils
import NLTE.collision_rates
import NLTE.NLTE_model
import utils
from line_formation import LineTransition, Photosphere
from NLTE.NLTE_model import (CollisionProcess, Environment,
                             HotElectronIonizationProcess, NLTESolver,
                             PhotoionizationProcess, RadiativeProcess,
                             RecombinationProcess, States)
from pcygni_5_Spec_Rel import PcygniCalculator

# truncate all data above this level (for now)
MAX_ENERGY_LEVEL = 25_000.0 # cm-1, corresponds to 400 nm from ground state.
# other than Sr II
ionization_stages_names = ['Sr I', 'Sr III', 'Sr IV', 'Sr V']

# strontium mass fraction (not just Sr II, all stages)
mass_fraction = 0.0002 # for initialization, will be fitted later

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
telluric_cutouts_albert = np.array([
    #[3000, 4500], # that's an absorption region
    #[5330, 5740], # I don't know why this is cut out from t=1.43d
    [6790, 6970], # NOTE: I chose this by visually looking at the spectra
    #[7070, 7300], # same as above
    [7490, 7650], # also same; NOTE: remember that telluric subtraction can be wrong
    [8850, 9700],
    [10950, 11600],
# I have been a bit generous in the choice, although the subtraction distorts the continuum
# in a broader range than this

    # [9940, 10300], # why was this masked?
    [12400, 12600],
    [13100, 14950], # was 14360
    [17550, 20050] # was 19000
])

all_masked_regions = np.append(telluric_cutouts_albert,[
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


def blackbody_flux(wav0, T, amplitude, z, disable_units=False):
    wav = wav0*(1+z)
    flux = amplitude * 2*h*c**2/(wav**5) * (1 / (np.exp(h*c/(wav*kB*T)) - 1))#)
    if disable_units == False:
        flux = flux.to('erg s-1 cm-2 AA-1')
    return flux


def pcygni_interp(wav_to_interp_at, v_max, v_phot, tau, resonance_wav, vref=0.22, ve=0.2, t_0=(1.43*u.day).to('s')):
    wav_grid, pcygni_profile = PcygniCalculator(t=t_0, vmax=v_max * c,
                                 vphot=v_phot * c, tauref=tau, vref=vref *c,
                                 ve=ve * c, lam0=resonance_wav).calc_profile_Flam(npoints=100, mode='both')
    # the PCygni calculator evaluated the profile at certain points it decided
    # we need to interpolate between these to obtain values we want to plot the profile at
    interpolator = interp1d(wav_grid, pcygni_profile, bounds_error=False, fill_value=1)
    return interpolator(wav_to_interp_at)


def guess_mass_frac(obs_spectrum, absorption_region, tau_init):
    # the way I'm guessing is. At flux minimum I_min/I_0 = exp(-\tau).
    #  We can take the -ln of this to get required \tau
    # since \tau is prop to num density (which is prop to mass fraction),
    #  this gives a good guess for mass fraction
    left, right = absorption_region
    select_absorp = (obs_spectrum[:,0] < left) & (obs_spectrum[:,0] > right)
    absorption_part = obs_spectrum[select_absorp]
    # to try and guess 
    I_min = np.min(absorption_part[:,1]) # find the I_min/I_0 point in the flux
    tau_req = -np.log(I_min)
    return (tau_req/tau_init)*mass_fraction


def get_line_transition_data(tau_matrix, states, level_occupancy):
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
    #print("SHAPE OF THE TAU MATRIX", tau_matrix.shape)
    for i in range(tau_matrix.shape[0]):
        for j in range(tau_matrix.shape[0]):
            if i <= j: continue
            optical_depths.append(tau_matrix[i,j])
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
    return optical_depths, line_wavelengths, g_upper, g_lower, n_upper, n_lower


'''def composite_pcygni(wavelength_grid, mass_fraction, v_out, v_phot, vref, ve,
                          environment, states, level_occupancy, A_rates):
    """
    Given a tau matrix, evaluate a line
    NOTE: So far this doesn't allow the 400nm and 1um features to trace different velocity fields
    TODO: Taking the product is a slight simplification, treat that more properly.
    """
    line_profiles = [] 
    t, occupancies, tau_matrices = NLTE.NLTE_model.solve_NLTE_sob(environment, states, solver, mass_fraction)
    optical_depths, wavelengths = get_line_depths_wavelengths(tau_matrices[-1], states)
    for tau, resonance_wav in zip(optical_depths, wavelengths):
        line_profiles.append(pcygni_interp(wavelength_grid, v_max=v_out, v_phot=v_phot, tau=tau,
                     resonance_wav=resonance_wav, vref=vref, ve=ve, t_0=(epoch * u.day).to('s')))
    product = np.prod(np.array(line_profiles), axis=0)
    if np.any(np.isnan(product)) > 0:
        print("THE PRODUCT CONTAINS NANs")
    print("The line profile has NaN values in wavelength range\n" ,wavelength_grid[np.isnan(product)])
    fig, ax = plt.subplots(figsize=(6,6))
    for i, line in enumerate(line_profiles):
        ax.plot(wavelength_grid, line, label=f'{wavelengths[i]:.2f}')
    ax.plot(wavelength_grid, product, label='total', c='k')
    ax.legend()
    #plt.show()
    return product


def fit_spectrum(obs_spec, environment, states, solver, level_occupancy, blackbody_continuum, absorption_region):
    normed_spec = obs_spec
    normed_spec[:,1] /= blackbody_continuum.eval(wavelength_grid=obs_spec[:,0])
    normed_spec[:,2] /= blackbody_continuum.eval(wavelength_grid=obs_spec[:,0])

    print("Level occupancy:", type(level_occupancy), level_occupancy.shape)
    v_ejecta_max = 0.5 #c, to not allow the fitter to use anything outrageous
    #tau_matrix = NLTE.NLTE_model.pop_to_tau(environment, states, [level_occupancy],
    #							  get_process(solver, RadiativeProcess).A,
    #							  mass_fraction)
    #line_profiles = composite_pcygni(tau_matrix, states, v_out=, v_phot=, v_ref=,ve=)

    pcygni_model = lambda wavelength_grid, mass_fraction, v_out, v_phot, vref, ve: \
                          composite_pcygni(wavelength_grid, mass_fraction, v_out=v_out, v_phot=v_phot, vref=vref, ve=ve,
                         environment=environment, states=states, 
            level_occupancy=level_occupancy,
              A_rates=get_process(solver, RadiativeProcess).A)
    #fixed_composite_pcygni = partial(composite_pcygni, )
    SpectrumModel = Model(pcygni_model)
    params = SpectrumModel.make_params(
        mass_fraction=dict(value=mass_fraction, max=1., min=1E-5),
        v_out=dict(value=0.2, max=v_ejecta_max, min=0.1, expr='v_out > v_phot'),
        v_phot=dict(value=0.2, max=v_ejecta_max, min=0.1),
        vref=dict(value=0.2, vary=False), # TODO: allow varying v_ref
        ve = dict(value=0.2, vary=False) # TODO: allow varying ve (scaling velocity)
    )
    valid_wavelengths = normed_spec[~np.isnan(normed_spec[:,0]),0]
    valid_fluxes = normed_spec[~np.isnan(normed_spec[:,0]),1]
    print("THIS WAS CROSSED!")
    return blackbody_continuum.eval(wavelength_grid=valid_wavelengths)*SpectrumModel.fit(valid_fluxes, params=params,
         wavelength_grid=valid_wavelengths, method='differential_evolution')'''


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
        for i, profile in enumerate(pcygni_profiles):
            ax2.plot(wavelength_grid, profile, label=f'{line_wavelengths[i].value:.2f}' + r'$\mathrm{\AA}$')
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

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from NLTE.NLTE_model import get_density_profile

M_ejecta = 0.04
atomic_mass = 88

n_e = lambda line_velocity, t_d, p=5, ve=0.284: (1.5e8*t_d**-3) * (line_velocity/ve)**-p
n_He = lambda line_velocity, t_d, mass_fraction=0.05: get_density_profile(M_ejecta, 4, mass_fraction)(line_velocity, t_d)
n_Sr = lambda line_velocity, t_d, mass_fraction: get_density_profile(M_ejecta, atomic_mass, mass_fraction)(line_velocity, t_d)
get_column_density = lambda mass_fraction, vphot, vmax, t_d: quad(n_Sr, vphot, vmax, args=(t_d, mass_fraction))[0]

'''def plot_velocity_shells(env):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define velocity range
    v_range = np.linspace(0.1, 0.9, 50)
    log_ne = np.log10(n_e(env.t_d, v_range))
    log_nSr = np.log10(n_Sr(v_range, env.t_d, env.mass_fraction))

    # Normalize values
    norm_ne = (log_ne - np.min(log_ne)) / np.ptp(log_ne)
    norm_nSr = (log_nSr - np.min(log_nSr)) / np.ptp(log_nSr)

    # Get colors and transparency
    colors = mpl.colormaps['plasma_r'](norm_ne[::-1])
    alphas = norm_nSr[::-1]  # Reverse to align with shell drawing

    # Draw shells
    patches = [Circle((0, 0), radius=v) for v in reversed(v_range)]
    collection = PatchCollection(patches, facecolor=colors, edgecolor=None, alpha=alphas)
    ax.add_collection(collection)

    # Draw photosphere velocity boundary
    ax.add_patch(Circle((0, 0), radius=env.line_velocity, edgecolor='k', fill=False, ls='--'))

    # Keep circles round
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')

    # Add labels
    ax.text(0, env.line_velocity + 0.05, ha='center', s=f'$v_{{phot}}={env.line_velocity:.2f}c$')
    ax.text(-0.63, 0.61, s=f'$t={env.t_d}$ days')

    ax.set_ylabel("$v$ [c]")
    # electron density colorbar
    sm_ne = mpl.cm.ScalarMappable(cmap='plasma_r', norm=mpl.colors.Normalize(vmin=np.min(log_ne), vmax=np.max(log_ne)))
    sm_ne.set_array([])
    cbar_ne = fig.colorbar(sm_ne, ax=ax, fraction=0.05, pad=0.02, extend='min')
    cbar_ne.set_label(r'$\log_{10} n_e$ [cm$^{-3}$]')
    cbar_ne.set_ticks(np.linspace(np.min(log_ne), np.max(log_ne), 4))
    cbar_ne.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in np.linspace(np.min(log_ne), np.max(log_ne), 4)])

    # draw a dashed line to mark the n_e of v_phot shell
    log_ne_phot = np.log10(n_e(env.t_d, env.line_velocity))
    normed_phot = (log_ne_phot - np.min(log_ne)) / np.ptp(log_ne)
    cbar_ne.ax.hlines(normed_phot * cbar_ne.ax.get_ylim()[1], xmin=0, xmax=1, color='k', linestyle='--', linewidth=1)

    # Transparency colorbar (horizontal, **above the figure**)
    single_color = mpl.colormaps['plasma_r'](0.25)  # Pick one color
    alpha_cmap = np.tile(single_color, (256, 1))
    alpha_cmap[:, -1] = np.linspace(0.1, 1, 256)  # Reverse for descending transparency
    alpha_cmap = mpl.colors.ListedColormap(alpha_cmap)

    sm_alpha = mpl.cm.ScalarMappable(cmap=alpha_cmap, norm=mpl.colors.Normalize(vmin=np.min(log_nSr), vmax=np.max(log_nSr)))
    sm_alpha.set_array([])

    # Move transparency colorbar **above** the figure
    cbar_alpha = fig.colorbar(sm_alpha, ax=ax, orientation='horizontal', fraction=0.05,# location='top',
                 pad=0.09, extend='max')
    cbar_alpha.set_label(r'$\log_{10}n_{Sr}$ [cm$^{-3}$]')
    cbar_alpha.set_ticks(np.linspace(np.min(log_nSr), np.max(log_nSr), 4))
    cbar_alpha.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in np.linspace(np.min(log_nSr), np.max(log_nSr), 4)])

    plt.savefig(f"velocity_shells_t{env.t_d:.2f}d.png", dpi=300, bbox_inches='tight')
    plt.show()'''



def compare_ionization_profile(v_shells, shell_occupancies, T, epoch):
    """
    Compare the ionization profile you get from nLTE vs LTE
    """
    ion_states = ['Sr I', 'Sr II', 'Sr III', 'Sr IV', 'Sr V']
    sr_ion_energies = np.array([5.70, 11.03, 42.88, 56.28]) * u.eV

    shell_occupancies = np.array(shell_occupancies)
    lte_ion_fractions = []
    # difference between velocity of the particular shell, and the v_phot
    delta_v = lambda v: v - v_shells[0]
    T_phot = lambda T, v: T/(1/np.sqrt(1 - delta_v(v)**2) * (1+delta_v(v)))
    for v in v_shells:
        lte_ion_fractions.append(compute_LTE_ionizations(n_e(v, epoch, p=5, ve=0.284),
                                 T_phot(T, v), sr_ion_energies, ion_states))
    lte_ion_fractions = np.array(lte_ion_fractions)

    srii_fracs = shell_occupancies[:, :len(SrII_states.names)].sum(axis=1)\
                        .reshape((len(v_shells), 1))
    
    nlte_fractions = np.append(srii_fracs, shell_occupancies[:,len(SrII_states.names):], axis=1)
    swap_nlte_fracs = nlte_fractions
    swap_nlte_fracs[:,1] = nlte_fractions[:,0]
    swap_nlte_fracs[:,0] = nlte_fractions[:,1]
    # how much charge does Sr I, Sr II, etc. each contribute to the charge state of the ejecta?
    ion_charges = np.linspace(0, 4, 5) # Sr V is +4
    mean_charge_lte = np.dot(lte_ion_fractions, ion_charges)
    mean_charge_nlte = np.dot(swap_nlte_fracs, ion_charges)

    f, a = plt.subplots(1,3, figsize=(10,5))
    colors = mpl.colormaps['viridis'](np.linspace(0, 1., len(ion_states)))
    for i, ion in enumerate(ion_states):
        a[0].plot(v_shells, lte_ion_fractions[:,i], ls='--', c=colors[i], label=ion + ' LTE')
        a[1].plot(v_shells, swap_nlte_fracs [:,i], ls='-', label=ion, c=colors[i])
        a[0].set_ylabel("Fraction")

    #print(f"For t={epoch}")
    a[0].set_title("LTE")
    a[1].set_title("nLTE")
    a[-1].plot(v_shells, mean_charge_lte, ls='--', label='LTE')
    a[-1].plot(v_shells, mean_charge_nlte, ls='-', label='NLTE')
    a[-1].set_ylabel("Mean Charge State")
    for ax in a:
        ax.legend(loc='upper right')
    #for i,j in np.ndindex((2,3)):
    #    a[i,j].set_xlabel("$v$ $[c]$")
    plt.suptitle(f"$t={epoch}$ days, " + r'\quad $\rho\propto (v/0.2)^{-5}$' + r'\quad $n_e=3\times 10^8(t/t_d)^{-3}(v/0.2)^{-5}$')
    plt.tight_layout()
    plt.show()



def plot_mean_ionization(v_shells, shell_occupancies, epoch):
    ne_p5_v284 = n_e(v_shells, t_d=epoch, p=5, ve=0.284)
    ne_p5_v2 = n_e(v_shells, t_d=epoch, p=5, ve=0.2)
    ne_p1_v2 = n_e(v_shells, t_d=epoch, p=1, ve=0.2)
    natoms = n_Sr(v_shells, t_d=epoch, mass_fraction=1)
    # level occupancies in each velocity shell
    shell_occupancies = np.array(shell_occupancies)
    # sum sr II bound states to get total Sr II
    srii_in_shells = shell_occupancies[:,:len(SrII_states.names)].sum(axis=1).reshape((len(v_shells),1))
    # array of Sr II, I, III .. etc. ionization fractions
    shell_ionizations = np.append(srii_in_shells, shell_occupancies[:,len(SrII_states.names):], axis=1)
    ion_charges = np.array([1,0,2,3,4]) # charge on Sr II, I, III, IV, V
    shell_mean_charge = np.array([ion_charges * shell for shell in shell_ionizations]).sum(axis=1)
    
    mean_charge_p1_v02 = [3.39098429, 3.40459839, 3.41766404, 3.43021164, 3.44226936, 3.45386402,
 3.46502015, 3.47576068, 3.48610741, 3.49608028 ,3.5056982 , 3.51497882,
 3.52393876, 3.53259342, 3.54095753, 3.54904479 ,3.55686807, 3.56443952,
 3.57177054, 3.578872  ]
    mean_charge_p5_v0284 = [3.21, 3.29, 3.36, 3.42, 3.48, 3.54, 3.58,3.628,
                         3.66,3.697,3.72,3.75306855,3.77,3.79,
                         3.81,3.831,3.84,3.85,3.87,3.88225936]
    natom_p5=[8657282,7670432,6815541,6072476,5424537,4857816, 4360679,3923361,
 3537630,3196525, 2894136, 2625433,
 2386118,2172508.54672726, 1981440.28366871, 1810186.42447293,
 1656390.63760226, 1518011.45866953 ,1393275.73851107, 1280639.61426591]
    mean_charge_p5_v02 = [3.8639471,  3.87991207, 3.89367332, 3.90557557, 3.91590459, 3.92489778,
 3.93275296, 3.93963527, 3.94568314, 3.95101295, 3.95572285, 3.9598959,
 3.96360263, 3.96690313, 3.96984875 ,3.97248351 ,3.97484527, 3.97696666,
 3.97887591, 3.9805975 ]
    print("mean ionization state, p=1", shell_mean_charge)
    print("atomic density for p=1", natoms)
    _, axis = plt.subplots(figsize=(6,6))

    axis.plot(v_shells, (ne_p5_v284/natom_p5)/mean_charge_p5_v0284,  ls=':', c='g',label="$n_e\propto (v/0.284)^{-5}$")
    axis.plot(v_shells, (ne_p5_v2/natom_p5)/mean_charge_p5_v02, ls='--', c='g',label="$n_e\propto (v/0.2)^{-5}$")
    axis.plot(v_shells, (ne_p1_v2/natoms)/mean_charge_p1_v02, ls='-', c='g',label="$n_e\propto (v/0.2)^{-1}$")
    axis.axhline(y=1, xmin=0, xmax=1, ls='--', lw=0.75, c='k')
    axis.text(y=1.05, x=0.34, s='(quasi)neutral plasma')
    axis.fill_between(v_shells, np.ones_like(v_shells)-0.025, np.ones_like(v_shells)+0.025, fc='dimgray',
                   alpha=0.3, zorder=-1)
    #axis.plot(v_shells, shell_mean_charge, ls='-', c='k', label=r"$n_e = 3\times10^8$")
    '''axis.plot(v_shells, mean_charge_p1_v02, ls='-', c='deeppink',label="mean charge state, $n_e\propto (v/0.2)^{-1}$")
    axis.plot(v_shells, mean_charge_p5_v0284, ls=':', c='deeppink',label="mean charge state, $n_e\propto (v/0.284)^{-5}$")
    axis.plot(v_shells, mean_charge_p5_v02, ls='--', c='deeppink',label="mean charge state, $n_e\propto (v/0.2)^{-5}$")'''
    axis.legend(loc='upper right', title='electron excess per ion')
    axis.set_xlabel("$v$ [c]")
    axis.set_title(r"$n_e=1.5 \times 10^8$ cm$^{-3}$ at $t=1$ day; $M_{ej}=0.04$M$_{\odot}$, $\mu=100$amu")
    plt.show()

def plot_velocity_shells(time_epochs, v_phots, v_maxes, mass_fractions):
    v_range = np.linspace(0.1, 0.9, 50)

    # Get min/max across all epochs for consistent colorbar scaling
    all_log_ne = [np.log10(n_e(v_range, t_d)) for t_d in time_epochs]
    all_log_nSr = [np.log10(n_Sr(v_range, t_d, mass_fraction)) for (t_d, mass_fraction) in zip(time_epochs, mass_fractions)]

    vmin_ne, vmax_ne = np.min(all_log_ne), np.max(all_log_ne)
    vmin_nSr, vmax_nSr = np.min(all_log_nSr), np.max(all_log_nSr)

    for t_d, v_phot, v_max, mass_fraction in zip(time_epochs, v_phots, v_maxes, mass_fractions):
        fig, ax = plt.subplots(figsize=(6, 6))

        log_ne = np.log10(n_e(v_range, t_d))
        log_nSr = np.log10(n_Sr(v_range, t_d, mass_fraction))

        # for a colorbar find a normalized scale; although maybe one doesn't have to do it this way
        norm_ne = (log_ne - vmin_ne) / (vmax_ne - vmin_ne)
        norm_nSr = (log_nSr - vmin_nSr) / (vmax_nSr - vmin_nSr)

        color_map = 'magma_r'
        colors = mpl.colormaps[color_map](norm_ne[::-1])
        alphas = norm_nSr[::-1]

        # Draw shells
        patches = [Circle((0, 0), radius=v) for v in reversed(v_range)]
        collection = PatchCollection(patches, facecolor=colors, edgecolor=None, alpha=alphas)
        ax.add_collection(collection)

        # draw ejecta extend (velocity boundaries)
        ax.add_patch(Circle((0, 0), radius=v_phot, edgecolor='k', fill=False, ls='--'))
        ax.add_patch(Circle((0,0), radius=v_max, ls=':', edgecolor='k', alpha=0.4, lw=0.75, fill=False))
        # fixed aspect ratio 
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')

        ax.text(0, v_phot + 0.05, ha='center', s=f'$v_{{phot}}={v_phot:.2f}c$')
        ax.text(-0.63, 0.61, s=f'$t={t_d}$ days')

        # color by electron density
        sm_ne = mpl.cm.ScalarMappable(cmap=color_map, norm=mpl.colors.Normalize(vmin=vmin_ne, vmax=vmax_ne))
        sm_ne.set_array([])
        cbar_ne = fig.colorbar(sm_ne, ax=ax, fraction=0.05, pad=0.02, extend='min')
        cbar_ne.set_label(r'$\log_{10}n_e$ [cm$^{-3}$]')
        cbar_ne.set_ticks(np.linspace(vmin_ne, vmax_ne, 4))
        cbar_ne.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in np.linspace(vmin_ne, vmax_ne, 4)])

        # draw a black line to mark what the n_e at v_phot is
        log_ne_phot = np.log10(n_e(v_phot, t_d))
        normed_phot = (log_ne_phot - vmin_ne) / (vmax_ne - vmin_ne)
        cbar_ne.ax.hlines(normed_phot * cbar_ne.ax.get_ylim()[1], xmin=0, xmax=1, color='k', linestyle='--', linewidth=1)

        # transparency colorbar for number density
        single_color = mpl.colormaps[color_map](0.6)
        alpha_cmap = np.tile(single_color, (256, 1))
        alpha_cmap[:, -1] = np.linspace(0, 1, 256)
        alpha_cmap = mpl.colors.ListedColormap(alpha_cmap)


        sm_alpha = mpl.cm.ScalarMappable(cmap=alpha_cmap, norm=mpl.colors.Normalize(vmin=vmin_nSr, vmax=vmax_nSr))
        sm_alpha.set_array([])
        cbar_alpha = fig.colorbar(sm_alpha, ax=ax, orientation='horizontal', fraction=0.05, pad=0.12, extend='max')
        cbar_alpha.set_label(r'$\log_{10}n_{Sr}$ [cm$^{-3}$]`')
        cbar_alpha.set_ticks(np.linspace(vmin_nSr, vmax_nSr, 4))
        cbar_alpha.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in np.linspace(vmin_nSr, vmax_nSr, 4)])

        # Mark number density at photosphere in transparency colorbar
        log_nSr_phot = np.log10(n_Sr(v_phot, t_d, mass_fraction))
        normed_nSr_phot = (log_nSr_phot - vmin_nSr) / (vmax_nSr - vmin_nSr)
        cbar_alpha.ax.vlines(normed_nSr_phot * cbar_alpha.ax.get_xlim()[1], ymin=0, ymax=1, color='k', linestyle='--', linewidth=1)

        plt.savefig(f"velocity_shells_t{t_d:.2f}d.png", dpi=300, bbox_inches='tight')
        plt.show()

def display_time_solution(t, level_occupancies, tau_all_timesteps, environment):
    # plot how the optical depth and populations achieve, and the ionization balance
    fig2, axes = plt.subplots(1, 3, figsize=(15,6))
    line_wavelengths = []
    for i in range(len(SrII_states.names)):
        for j in range(len(SrII_states.names)): 
            if i <= j: continue
            wavelength = np.abs(SrII_states.energies[i] - SrII_states.energies[j]).to('AA', equivalencies=u.spectral())
            line_wavelengths.append(wavelength)
            axes[0].plot(t, tau_all_timesteps[:,i,j], label=f"{wavelength.value:.1f}" + "$\mathrm{\AA}$")
                 #label=SrII_states.tex_names[i] + r'$\to$ ' + SrII_states.tex_names[j])
    axes[0].set_ylabel(r'Sobolev Optical Depth [$\tau$]')
    axes[0].text(x=3e-1, y=5*np.max(tau_all_timesteps[-1,:,:]), s=f'$t_d={environment.t_d}$ days',va='bottom', ha='left')
    
    # also plot the nLTE populations
    level_colors = mpl.colormaps['plasma'](np.linspace(0, 1, len(SrII_states.names)))
    # and for comparison, LTE populations
    LTE_pops = get_LTE_pops(SrII_states, environment.T_electrons)
    # estimate the Sr II fraction.
    srii_frac = level_occupancies[:len(SrII_states.names)].sum(axis=0)/level_occupancies.sum(axis=0)
    for i in range(len(SrII_states.names)):
        axes[1].plot(t, level_occupancies[i]/srii_frac,
                            label=SrII_states.tex_names[i], c=level_colors[i])
        axes[1].axhline(y=LTE_pops[i], xmin=0, xmax=1, ls='--', c=level_colors[i])
    #axes[1].axhline(y=1, xmin=0, xmax=1, c='k', ls='-')
    axes[1].set_ylabel('Level Occupancy')
    
    scientific_notation = lambda num: r"{:.2f} \times 10^{}".format(num / 10**int(np.log10(num)), int(np.log10(num)))
    axes[1].text(x=5e-11, y=0.5, s=f'$T_e={environment.T_electrons:.0f}$K')
    axes[2].text(x=5e-11, y=0.4, va='top', s='$n_e=' + scientific_notation(environment.n_e) +'$ [cm$^{-3}]$')
    # NOTE: the pop_to_tau method alone doesn't take escape probability into account; maybe adjust
    #LTE_opacities = NLTE.NLTE_model.pop_to_tau(np.array([LTE_pops]))
    srII_ion_stages = np.append(
                                [level_occupancies[:len(SrII_states.names), :].sum(axis=0)], # sum individual Sr II levels for total Sr II
                                level_occupancies[len(SrII_states.names):, :], # remaining Sr I, III, IV, ..
                         axis=0)
    ion_stage_names = ['Sr II'] + ionization_stages_names
    axes[2].set_ylabel("Ionization Fraction")
    axes[2].set_yscale('log')
    ion_stage_colors = mpl.colormaps['plasma'](np.linspace(0, 1, len(ion_stage_names)+1))

    lte_ionization = compute_LTE_ionizations(environment.n_e, environment.T_electrons,
                                    np.array([5.70, 11.03, 42.88, 56.28]) * u.eV,
                                        ['Sr I', 'Sr II', 'Sr III', 'Sr IV', 'Sr V'])
    print("Ionization Fractions in LTE are:", lte_ionization)
    # swap 'Sr I' and 'Sr II' in this because.. that's just how the 'occupancies' vector is arranged.
    lte_ion_swapped = lte_ionization
    lte_ion_swapped[1] = lte_ionization[0]
    lte_ion_swapped[0] = lte_ionization[1]
    # now plot
    for i, (ion, name) in enumerate(zip(srII_ion_stages, ion_stage_names)):
        axes[2].plot(t, ion, label=name + f' ({100*ion[-1]:.2f}\%)', c=ion_stage_colors[i])
        axes[2].axhline(y=lte_ion_swapped[i], xmin=0, xmax=1, ls='--', c=ion_stage_colors[i])
    for ax in axes:
        ax.legend(loc='lower left')
        ax.set_xlabel("Time")
        ax.set_yscale("log")
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"t{environment.t_d}d_time_solution.png",dpi=300)


def plot_tau_shells(tau_shells, tau_wavelengths):
            # from a previous calculation, for v^-5
        '''tau_p5 = [9.114015709300082, 7.217711768792224, 5.685746712516514, 4.4509136370419995, 3.4543294432319516,
             2.6485159771693207, 1.9977375671754007, 1.4762030727832955, 1.0650312744635027, 0.748982447996239,
               0.5138129304767134, 0.3454028661442654, 0.22859733221609924, 0.1496876704097265, 0.0974810736979666,
             0.06340047750469706, 0.04130854876669985, 0.027019762770838323, 0.017767115697160718, 0.011754901118656805]'''
        tau_shells = np.array(tau_shells)[:,3:9]
        print("Wavelengths", tau_wavelengths)
        tau_wavelengths = np.array([t.value for t in tau_wavelengths])[3:9] * u.AA
        fig_tau, tau_ax = plt.subplots(2,3, figsize=(8,6))
        for i, j in np.ndindex((2,3)):
            #if 3*i+j >=5: continue
            #tau_ax.plot(v_shells, tau_p5, ls='-', c='lightpink', alpha=0.6, label= 'NLTESolver ' + r"$\tau_{\textrm{sob}}$ " + f'$({tau_wavelength.value:.1f}\AA)$ ' + r'$\rho\sim v^{-5}$')
            tau_ax[i,j].plot(v_shells, tau_shells[:,3*i+j], ls='-', c='lightgreen', label=f'$({tau_wavelengths[3*i+j].value:.1f}\AA)$ ')
                     #label= 'NLTESolver ' + r"$\tau_{\textrm{sob}}$" + f'$({tau_wavelength[i+j].value:.1f}\AA)$ ' + r'$\rho\sim v^{-1}$')
            tau_ax[i,j].plot(v_shells, tau_v(v_shells, vphot=v_phots[0], tauref=tau_shells[0,3*i+j], ve=0.32), ls='-.', 
                                c='dimgray')#, label=r"$\tau =\tau_{\mathrm{ref}}e^{(v_{\mathrm{phot}}-v)/v_e}$ for $v_e=0.32c$")
            tau_ax[i,j].plot(v_shells, tau_v(v_shells, vphot=v_phots[0], tauref=tau_shells[0,3*i+j], ve=1.), ls=':',
                                        c='dimgray')#, label=r"$\tau =\tau_{\mathrm{ref}}e^{(v_{\mathrm{phot}}-v)/v_e}$ for $v_e=1c$")
            tau_ax[i,j].legend()
            #print("Values of tau in the shells for p=-5\n", tau_shells)
            tau_ax[i,j].set_xlabel("$v$ [$c$]")
            tau_ax[i,j].set_ylabel(r"$\tau(v)$")
        plt.tight_layout()
        #plt.savefig('tau_sob_vshells_p-1.png', dpi=300)
        plt.show()

def plot_electron_density(t_d, v_phot, v_out):
    v = np.linspace(0.05, 0.5)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(v, n_e(v, t_d), label='electron density', c='red')
    ax.plot(v, n_Sr( v, t_d,1), label=r'all atoms $\langle A\rangle = 88$', c='black', ls='-')
    #ax.plot(v, n_He( v, t_d,0.05), label='helium $X(He)=0.05$',c='black', ls=':')
    #ax.plot(v, n_Sr(v, t_d, 0.95) + n_He(v, t_d, 0.05), label='nucleus per cm$^{-3}$')
    print("number density of all atoms at photosphere\n", n_Sr(v_phot, t_d, 1))
    print("electron density at photosphere: ", n_e(v_phot, t_d))
    print("at t_d, if strontium was 0.1%, ", n_Sr(v_phot, t_d, mass_fraction=0.001))
    ax.axvline(v_phot, ls='--', c='dimgray', lw=0.75,ymin=0, ymax=1)
    ax.set_ylabel("$n$ [cm$^{-3}$]")
    ax.set_yscale('log')
    ax.set_xlabel("$v$ [c]")
    ax.legend(loc='upper right')
    plt.savefig('ejecta_density_profile_t143.png',dpi=300)
    plt.show()

def plot_ionization_temporal(T, epochs, v_phots):
    ne_emergence = lambda line_velocity, t_d, p=5, ve=0.284: (1e8*(t_d/1.5)**-3) * (line_velocity/ve)**-p
    ion_fractions = []
    ion_states = ['Sr I', 'Sr II', 'Sr III', 'Sr IV', 'Sr V']
    sr_ion_energies = np.array([5.70, 11.03, 42.88, 56.28]) * u.eV
    # doppler shifted temperature
    T_rad = lambda T, v, v_phot: T/(1/np.sqrt(1 - (v-v_phot)**2) * (1+(v-v_phot)))

    # spatial profile of ionization at t=1.43 days
    v_shells = np.linspace(0.2, 0.5, 100)
    # power law electron density, doppler corrected temperature
    LTE_ions_shell = []
    for v in v_shells:
        LTE_ions_shell.append(compute_LTE_ionizations(ne_emergence(v, epochs[0]), T_rad(T[0], v, v_phots[0]),
                                          ionization_energies= sr_ion_energies, species=ion_states))
    LTE_ions_shell = np.array(LTE_ions_shell)
    for i in range(len(epochs)):
        ne = n_e(line_velocity=v_phots[i], t_d=epochs[i], p=5, ve=0.284)
        ion_fractions.append(compute_LTE_ionizations(ne_emergence(v_phots[i], epochs[i]), T[i],
                            ionization_energies=sr_ion_energies,
                            species=ion_states))
    ion_fractions = np.array(ion_fractions)
    mean_ion_state = np.dot(ion_fractions, np.linspace(0, 4, 5))
    f, a = plt.subplots(1, 3, figsize=(10,4))
    colors = mpl.colormaps['viridis'](np.linspace(0, 1., len(ion_states)))#[n]
    for i, ion in enumerate(ion_states):
        print(f"Ionization fractions for {ion} across epochs", ion_fractions[:,i])
        a[0].plot(v_shells, LTE_ions_shell[:,i], c=colors(len(ion_states))[i], label=ion)
        a[1].plot(epochs, ion_fractions[:,i], c=colors(len(ion_states))[i], ls='--', label=ion_states[i])
    a[2].plot(epochs, mean_ion_state, ls='--', label="mean ionization")
    a[1].set_xlabel("Time [days]")
    a[2].set_xlabel("Time [days]")
    a[0].set_ylabel("Fraction")
    a[1].set_ylabel("Fraction")
    a[0].set_xlabel("velocity [c]")
    a[0].text(x=0.2, y=0.95, s='$t=1.43$ days')
    #a[2].set_ylabel("Fraction")
    for ax in a:
        ax.legend()
    plt.tight_layout()
    plt.show()	


def print_luminosities(states, line_luminosities):
    wavelengths = []
    luminosities = []
    for i in range(len(states.names)):
        for j in range(len(states.names)):
            dE = states.energies[i] - states.energies[j]
            if dE >= 0: continue
            wavelengths.append(dE.to('nm', equivalencies=u.spectral()))
            luminosities.append(line_luminosities[i,j])
    for line, lumi in zip(wavelengths, luminosities):
        print(f"Luminosity of {line:.1f}nm line: ", lumi)

def print_line_things(states, v_shells, shell_occupancies, tau_shells):
    v_shells = np.array(v_shells)
    tau_shells = np.array(tau_shells)
    shell_occupancies = np.array(shell_occupancies)
    x = 0
    ve = 0.32
    with open("dump.txt", "w") as f:
        f.write("velocity grid\n" + str(list(v_shells)) + '\n')
        W = lambda v: 0.5*(1- np.sqrt(1 - (v_shells[0]/v)**2))
        fi, axi = plt.subplots(figsize=(6,6))
        colors = mpl.colormaps['Spectral'](np.linspace(0, 1, tau_shells.shape[1]))
        for i in range(len(states.names)):
            for j in range(len(states.names)):
                if i <= j: continue
                #if x <=4 or x>=7: continue
                if states.energies[i] > states.energies[j]:
                    g_u = states.multiplicities[i]
                    g_l = states.multiplicities[j]
                    n_l = shell_occupancies[:,j]
                    n_u = shell_occupancies[:,i]
                else:
                    g_l = states.multiplicities[i]
                    g_u = states.multiplicities[j]
                    n_u = shell_occupancies[:,j]
                    n_l = shell_occupancies[:,i]
                lamb = (states.energies[i] - states.energies[j]).to('nm', equivalencies=u.spectral())
                nu = lamb.to("Hz", equivalencies=u.spectral())
                #print("Wavelength:", lamb, f"\n tau:{list(tau_shells[:,x])}\n occupancies = ")
                S = (2*h*nu**3/(c**2))*(g_u*n_l/(g_l*n_u) -1)**-1 / u.sr
                B_lamb = np.array([BlackBody(T_e * u.K, scale=W(v)*u.Unit("erg/(sr cm2)"))(nu).value for v in v_shells]) * u.Unit("erg / (cm2 sr)")
                #print("Values of the Planck function", B_lamb.to("erg/cm^2"))
                I_emit = B_lamb*np.exp(-tau_shells[:,x]) + S*(1-np.exp(-tau_shells[:,x]))
                I_emit_dil = B_lamb*np.exp(-tau_shells[:,x]) + W(v_shells)*(1-np.exp(-tau_shells[:,x]))*B_lamb
                tau_exp = tau_shells[0,x]*np.exp((v_shells[0]- v_shells)/ve)
                I_emit_exp = B_lamb*np.exp(-tau_exp) + W(v_shells)*(1-np.exp(-tau_exp))*B_lamb
                f.write(f"For line {lamb.value:.2f}\n")#, source function:\n", list(S))
                f.write("optical depths\n" + str(list(tau_shells[:,x]))+ "\n")
                #print("Emergent intensities\n",I_emit)
                if np.isclose(lamb.to("nm").value, 1003.9, rtol=1e-2):
                    axi.plot(v_shells, I_emit, ls='-', label=f'{lamb.value:.1f}nm ' + r'$S(p,z) \propto \left[\left(\frac{g_u n_l}{g_l n_u} -1 \right) \right]^{-1}$, $\tau$ explicit', c=colors[x])
                    axi.plot(v_shells, I_emit_dil, ls='-.', label=f'{lamb.value:.1f}' + r'nm $S=W(v)I$, $\tau \propto (v/0.2)^{-5}$, $\tau$ explicit', c=colors[x])
                    axi.plot(v_shells, I_emit_exp, ls=':', label=f'{lamb.value:.1f}' + r'nm $S=W(v)I$, $\tau \propto e^{(v_{phot}-v)/v_e}$', c='k')
                x+=1
                #break
            #
            #    break
        hand, labels = axi.get_legend_handles_labels()
        axi.legend(loc='upper right')
        axi.set_xlabel("$v[c]$")
        axi.set_ylabel(r"$I_{emit}(" + f'{1003.9}'+r"\mathrm{nm}) = B_{\lambda}e^{-\tau} + S(1-e^{-\tau})$")
        axi.set_title(f"Emergent specific intensity beam of wavelength $\lambda_0$, t={epoch} days")
        #plt.show()


absorption_region = (7000, 9700)
if __name__ == "__main__":
    #plot_electron_density(t_d=1.43, v_phot=0.235, v_out=0.6)
    SrII_states = get_strontium_states()
    SrII_states.texify_names() # TODO: make this get called automatically post-init in NLTE_model.py

    fig, ax = plt.subplots(figsize=(8,6))

    # put the telluric masks
    flux_min_grid = -5.5E-16 * np.ones(100)
    flux_max_grid = 4E-16 * np.ones(100)
    for (left, right) in telluric_cutouts_albert:
        horizontal_grid = np.linspace(left, right, 100)
        ax.fill_between(horizontal_grid, flux_max_grid, flux_min_grid, fc='silver', alpha=0.25)
  
    # now plot the spectra, blackbody + pcygni fits
    wavelength_grid = np.linspace(2500, 23000, 10_000) * u.AA

    T_elec_epochs = {1.43: 4400,
                    2.42: 3200,
                    3.41: 2900,
                    4.40: 2800}
    
    offsets = np.array([0., -2., -3.5, -5])*1E-16
    scale = np.array([1.2, 2.5, 3., 3.5])
    v_outs = [0.425, 0.35, 0.286, 0.271]
    v_phots = [0.29, 0.15, 0.2, 0.17]#[0.236, 0.19, 0.18, 0.162]#[0.2, 0.13, 0.12, 0.11]
    ve_s = [0.32] * 4 
    v_refs = [0.2] * 4
    mass_fractions = [0.001, 0.005, 0.30, 1.]#[0.00045, 0.03, 0.15, 0.4]#[0.00055, 0.0015, 0.0075, 0.007]#[0.00008, 0.00004, 0.00015, 0.0002]

    aayush_colors = mpl.colormaps['Spectral'](np.linspace(0, 1., len(T_elec_epochs)))
    #plot_ionization_temporal(np.array([4400, 3200, 2900, 2800]), np.array([1.43, 2.42, 3.41, 4.40]),
    #                       np.array(v_phots))
    #tau_wavelength = None
    tau_v = lambda v, vphot, tauref, ve: tauref*np.exp((vphot-v)/ve)
    #plot_velocity_shells(T_elec_epochs.keys(), v_phots, v_outs, mass_fractions)
    for i, (epoch, T_e) in enumerate(T_elec_epochs.items()):
        tau_shells = []
        v_shells = np.linspace(v_phots[i], v_outs[i], 20)
        print("COUNTING!: ", i, epoch, T_e)
        colors = mpl.colormaps['coolwarm'](np.linspace(0, 1., len(T_elec_epochs)))
        spec_ep = xshooter_data(day=epoch)
        # sets the amplitude, etc. of the fitted blackbody 
        fitted_cont = utils.fit_blackbody(spec_ep[:,0], spec_ep[:,1], all_masked_regions)
        # only thing that needs to be tuned is then the mass_fraction
        #fitted_spectral_line = utils.fit_planck_with_pcygni(spec_ep[:,0], spec_ep[:,1], telluric_cutouts_albert)
        T = fitted_cont.params['T'].value
        T_sigma = fitted_cont.params['T'].stderr
        amplitude = fitted_cont.params['amplitude'].value
        
        # this continuum object will be passed to the line formation code.
        continuum = BlackBody(temperature = T * u.K, scale=amplitude*u.Unit("1e20 * erg/(s sr cm2 AA)"))
        
        shell_occupancies = []
        g_upper = []
        g_lower = []
        n_upper_grid = []
        n_lower_grid = []
        for v in v_shells:
            environment = Environment(t_d=epoch,
                                    T_phot=T_e,
                                    mass_fraction=mass_fractions[i],
                                    atomic_mass=88,
                                    photosphere_velocity=v_phots[i],
                                    line_velocity=v,#v_phots[i],
                                    T_electrons=T_e)
            solver = NLTESolver(environment, SrII_states, processes=
                                    [RadiativeProcess(SrII_states, environment),
                                    CollisionProcess(SrII_states, environment),
                                    HotElectronIonizationProcess(SrII_states, environment),
                                    RecombinationProcess(SrII_states, environment),
                                    #PhotoionizationProcess(SrII_states, environment)
                            ])
            #plot_velocity_shells(environment)
            print(f"Given mass fraction X={mass_fractions[i]}, n_Sr={environment.n_He} at t={epoch}")
            '''sr_coll_matr = get_rate_matrix(solver, CollisionProcess)
            recombination_matrix = get_rate_matrix(solver, RecombinationProcess)
            nonthermal_matrix = get_rate_matrix(solver, HotElectronIonizationProcess)
            
            utils.display_rate_timescale(recombination_matrix, SrII_states.tex_names + ionization_stages_names,
                                    'Recombination', environment)
            utils.display_rate_timescale(nonthermal_matrix, SrII_states.tex_names + ionization_stages_names,
                                    'Non-thermal Ionization', environment)'''

            # estimate non-LTE atomic populations, tau_all_timesteps, line_luminosities \
            t, level_occupancy, tau_matrix = NLTE.NLTE_model.solve_NLTE_sob(
                                                        environment, 
                                                        SrII_states, 
                                                        solver, 
                                                        mass_fractions[i]
                                                    )
            # use these populations and fit a spectrum
            '''fitted_spec = fit_spectrum(spec_ep, environment, SrII_states, solver,
                                        level_occupancy[:, -2:-1], # steady state level occupancy
                                        fitted_cont,
                                        absorption_region) # blackbody parameters
            x_sr = fitted_spec.pars['mass_fraction']#.value and .stderr '''
            line_depths, resonance_wavelengths, g_upper, g_lower, n_upper, n_lower = get_line_transition_data(tau_matrix, SrII_states, level_occupancy[:,-1])
            tau_shells.append(line_depths)
            #tau_wavelength = resonance_wavelengths # this should be the resonance line
            shell_occupancies.append(level_occupancy[:,-1])
            n_upper_grid.append(n_upper)
            n_lower_grid.append(n_lower)
        # plot the spatial profile of tau
        if False:
            plot_tau_shells(tau_shells, tau_wavelength)
            plot_mean_ionization(v_shells, shell_occupancies, epoch)
        print("At t=",epoch)
        #print_luminosities(SrII_states, line_luminosities, tau_shells)
        #print_line_things(SrII_states, v_shells, shell_occupancies, tau_shells)
        #compare_ionization_profile(v_shells, shell_occupancies, T, epoch)

        # convert these list objects to a numpy array
        tau_shells = np.array(tau_shells)
        n_upper_grid = np.array(n_upper_grid)
        n_lower_grid = np.array(n_lower_grid)
        #for grid in [tau_shells, n_upper_grid, n_lower_grid]:
        #    grid = np.array(grid)

        line_transition_objects = []
        for j, line_wavelength in enumerate(resonance_wavelengths):
            # ignore forbidden lines that are too weak and will not be visible
            if np.max(tau_shells[:,j]) < 1e-3: continue
            line_transition = LineTransition(
                                    line_wavelength,
                                    tau_shells[:,j],
                                    v_shells, 
                                    g_upper[j], 
                                    g_lower[j], 
                                    n_upper_grid[:,j], 
                                    n_lower_grid[:,j]
                            )
            line_transition_objects.append(line_transition)
        photosphere = Photosphere(v_phot=v_phots[i] * c, 
                                  v_max=v_outs[i] * c,
                                  t_d=epoch * u.day,
                                  continuum=continuum,
                                  line_list=line_transition_objects)
        
        full_wav_grid, spectrum_flux = photosphere.calc_spectrum()
        #exit()
        ax.plot(full_wav_grid, scale[i]*spectrum_flux + offsets[i], marker='.',
                            ls='--', ms=1, c=aayush_colors[i], label=f"more physical model t={epoch}")
        #    print(f"Line {res_lamb:.1f}nm
        pcygni_line = lambda wav, vphot=v_phots[i], vout=v_outs[i]: blackbody_with_pcygnis(wav, tau_shells[0,:], resonance_wavelengths,
                                    fitted_cont, t_0=(epoch * u.day).to('s'), v_out=vout, v_phot=vphot, ve=ve_s[i], display=False)
                            #environment=environment, states=SrII_states, level_occupancy=level_occupancy[:, -2:-1], A_rates=)
        ax.plot(wavelength_grid, scale[i]*pcygni_line(wavelength_grid) + offsets[i], c='k', ls='-', lw=0.25)
        ax.fill_between(wavelength_grid.value, scale[i]*pcygni_line(wavelength_grid) + offsets[i],
                scale[i]*fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
                            fc='#ebc3d4', alpha=0.7) #ebc3d4

        '''v_grid = np.linspace(v_phots[i]-0.03, v_phots[i]+0.03, 60)
        grid_residuals = []
        for v in v_grid:
            spectrum_residuals = np.abs(pcygni_line(spec_ep[:,0], vphot=v) - spec_ep[:,1])
            abs_part = (spec_ep[:,0] > absorption_region[0]) & (spec_ep[:,0] < absorption_region[1])
            grid_residuals.append(np.sum(spectrum_residuals[abs_part]))
        mass_residuals_list.append(grid_residuals)
        grid_2d = np.array(mass_residuals_list)
        p, q = np.unravel_index(grid_2d.argmin(), grid_2d.shape)
        print("For time", epoch, "Best combination: X(Sr)", mass_grid[p], 'v_phot', v_grid[q])
        print("Searched velocity grid\n", v_grid)
        print("Mass grid\n", mass_grid)
        f, a = plt.subplots()
        m = a.matshow(grid_2d, cmap='RdYlBu')
        a.set_xticks(v_grid, minor=True)
        a.set_yticks(mass_grid, minor=True)
        a.grid(which='minor', color='k', linestyle='-', linewidth=0.1)
        plt.colorbar(m, ax=a)'''
        #plt.show()
            #a.plot(v_grid, grid_residuals, 'bo-', label='$v_{phot}$ at $t_d=' + str(epoch) + '$')
            #ax.legend()
        #print(f"At t={epoch}, the residuals are minimum for {v_grid[np.argmin(grid_residuals)]}")
        #print("Fo")

        # place text stating X(Sr) at an appropriate height
        desired_text_loc = 4_000
        valid_flux = ~np.isnan(spec_ep[:, 1])  # Mask for valid flux values
        valid_indices = np.where(valid_flux)[0]  # Indices of valid flux
        closest_index = valid_indices[np.argmin(np.abs(spec_ep[valid_indices, 0] - desired_text_loc))]
        loc_flux = spec_ep[closest_index, 1]

        ax.text(x=desired_text_loc, y=offsets[i] + loc_flux - 0.4E-16, ha='left',
                      s='$X_{Sr}=' + f'{mass_fractions[i]*100:.3f} \%$', c=colors[i])
        ax.plot(wavelength_grid, scale[i]*fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
                       ls='-', color='dimgray', #label=f"{T:.2f} $\pm$ {T_sigma:.2f}",
                       lw=0.5,)
        ax.plot(spec_ep[:,0], scale[i]*spec_ep[:,2]+ offsets[i], ls='-', color='darkgray', alpha=0.4, lw=0.75) #/fitted_cont.eval(wavelength_grid=spec_ep[:,0]) - i 
        ax.plot(spec_ep[:,0], scale[i]*spec_ep[:,1]+ offsets[i], ls='-', color=colors[i], lw=0.75,
                          alpha=1., label=f'$t={epoch}$ days')
        # TODO: 
        #display_time_solution(t, level_occupancy, tau_all_timesteps, environment)
    ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
    ax.set_xlabel("Wavelength [$\mathrm{\AA}$]")
    ax.legend(loc='upper right')
    #ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('fitted_cpygni.png', dpi=300)
    plt.show()

import os
from pathos.multiprocessing import ProcessingPool
from collections.abc import Iterable
from fractions import Fraction
from functools import cache, partial

import time
import multiprocessing as mp
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

import utils.atomic_utils as atomic_utils
import misc
import NLTE.collision_rates
import NLTE.NLTE_model
import utils.utils as utils
from line_formation import LineTransition, Photosphere
from NLTE.NLTE_model import (CollisionProcess, Environment,
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


def pcygni_interp(wav_to_interp_at, v_max, v_phot, tau, resonance_wav, vref=0.22, ve=0.2, t_0=(1.43*u.day).to('s')):
    wav_grid, pcygni_profile = PcygniCalculator(t=t_0, vmax=v_max * c,
                                 vphot=v_phot * c, tauref=tau, vref=vref *c,
                                 ve=ve * c, lam0=resonance_wav).calc_profile_Flam(npoints=100, mode='both')
    # the PCygni calculator evaluated the profile at certain points it decided
    # we need to interpolate between these to obtain values we want to plot the profile at
    interpolator = interp1d(wav_grid, pcygni_profile, bounds_error=False, fill_value=1)
    return interpolator(wav_to_interp_at)


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

from NLTE.NLTE_model import get_density_profile

M_ejecta = 0.04
atomic_mass = 88



absorption_region = (7000, 9700)
if __name__ == "__main__":
    #misc.plot_electron_density(t_d=1.43, v_phot=0.235, v_out=0.6)
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
    v_phots = [0.29, 0.23, 0.2, 0.17]#[0.236, 0.19, 0.18, 0.162]#[0.2, 0.13, 0.12, 0.11]
    ve_s = [0.32] * 4 
    v_refs = [0.2] * 4
    mass_fractions = [0.3, 0.2, 0.3, 0.2]#[0.00045, 0.03, 0.15, 0.4]#[0.00055, 0.0015, 0.0075, 0.007]#[0.00008, 0.00004, 0.00015, 0.0002]

    aayush_colors = mpl.colormaps['Spectral'](np.linspace(0, 1., len(T_elec_epochs)))
    #misc.plot_ionization_temporal(np.array([4400, 3200, 2900, 2800]), np.array([1.43, 2.42, 3.41, 4.40]),
    #                       np.array(v_phots))
    #tau_wavelength = None

    def _compute_tau_shell_sr(v_line, epoch, T_elec, mass_fraction, atomic_mass,
                           v_phot):
        #print("Initializing for line_velicity", v_line)
        env = Environment(t_d=epoch, T_phot=T_elec, mass_fraction=mass_fraction,
                          atomic_mass=atomic_mass, photosphere_velocity=v_phot,
                          line_velocity=v_line, T_electrons=T_elec)

        processes = processes=[
                                RadiativeProcess(SrII_states, env),
                                CollisionProcess(SrII_states, env),
                                HotElectronIonizationProcess(SrII_states, env),
                                RecombinationProcess(SrII_states, env),
                                # PhotoionizationProcess(SrII_states, env)
                            ]
        #print(SrII_states.all_names)
        #print("Transition rate matrix for radiative process:\n", processes[3].get_transition_rate_matrix())
        #utils.display_rate_timescale(processes[0].get_transition_rate_matrix(), SrII_states.all_names,
        #                             process_name = processes[0].name, environment=env)

        solver = NLTESolver(env, SrII_states, processes=processes)
        t_arr, pops, tau_mat = NLTE.NLTE_model.solve_NLTE_sob(env, SrII_states, solver, mass_fraction)

        depths, wavelengths, g_up, g_lo, n_up, n_lo = get_line_transition_data(
            tau_mat, SrII_states, pops[:, -1]
        )
        return depths, pops[:, -1], g_up, g_lo, n_up, n_lo, wavelengths
    


    tau_v = lambda v, vphot, tauref, ve: tauref*np.exp((vphot-v)/ve)
    #misc.plot_velocity_shells(T_elec_epochs.keys(), v_phots, v_outs, mass_fractions)
    
    
    
    
    for i, (epoch, T_e) in enumerate(T_elec_epochs.items()):
        tau_shells = []
        v_shells = np.linspace(v_phots[i],0.5, 50)



        mp.context._force_start_method('spawn')
        compute_tau_shell_worker = partial(_compute_tau_shell_sr,
                                           epoch=epoch,
                                            T_elec=T_e,
                                            atomic_mass=88,
                                            mass_fraction=mass_fractions[i],
                                            v_phot=v_phots[i])
        with ProcessingPool(num_cores) as pool:
            results = pool.map(compute_tau_shell_worker, v_shells)


        # TODO: unpaack  the results here

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
        print("At t=",epoch)
        #misc.print_luminosities(SrII_states, line_luminosities, tau_shells)
        #misc.print_line_things(SrII_states, v_shells, shell_occupancies, tau_shells)
        #misc.compare_ionization_profile(v_shells, shell_occupancies, T, epoch)

        # convert these list objects to a numpy array
        tau_shells, shell_occ, g_upper, g_lower, n_up_grids, n_lower_grid, resonance_wavelengths = zip(*results)
        tau_shells   = np.array(tau_shells)
        shell_occ    = np.array(shell_occ)
        n_upper_grid   = np.array(n_up_grids)
        n_lower_grid   = np.array(n_lower_grid)
        #tau_shells = np.array(tau_shells)
        #n_upper_grid = np.array(n_upper_grid)
        #n_lower_grid = np.array(n_lower_grid)
        #for grid in [tau_shells, n_upper_grid, n_lower_grid]:
        #    grid = np.array(grid)

        #print(tau_shells[0,:])
        #exit()
        line_transition_objects = []
        for j, line_wavelength in enumerate(resonance_wavelengths[0]):
            # ignore forbidden lines that are too weak and will not be visible
            if line_wavelength.to("nm").value > 2_000: continue
            line_transition = LineTransition(
                                    (line_wavelength),#.to("cm").value,
                                    tau_shells[:,j], # equatorial ejecta
                                    tau_shells[:,j], # polar ejecta
                                    v_shells, 
                                    g_upper[0][j], 
                                    g_lower[0][j], 
                                    n_upper_grid[:,j], 
                                    n_lower_grid[:,j]
                            )
            line_transition_objects.append(line_transition)
        
        photosphere = Photosphere(v_phot=(v_phots[i] * c),#.cgs.value, 
                                  v_max=(v_outs[i] * c),#.cgs.value,
                                  t_d=(epoch * u.day),#.cgs.value,
                                  continuum=continuum,
                                  line_list=line_transition_objects)
        #photosphere.visualize_polar_region()
        full_wav_grid, spectrum_flux = photosphere.calc_spectrum()
        #exit()
        ax.plot(full_wav_grid, scale[i]*spectrum_flux + offsets[i], marker='.',
                            ls='--', ms=1, c=aayush_colors[i], label=f"more physical model t={epoch}")
        #    print(f"Line {res_lamb:.1f}nm
        #pcygni_line = lambda wav, vphot=v_phots[i], vout=v_outs[i]: blackbody_with_pcygnis(wav, tau_shells[0,:], resonance_wavelengths,
        #                            fitted_cont, t_0=(epoch * u.day).to('s'), v_out=vout, v_phot=vphot, ve=ve_s[i], display=False)
                            #environment=environment, states=SrII_states, level_occupancy=level_occupancy[:, -2:-1], A_rates=)
        #ax.plot(wavelength_grid, scale[i]*pcygni_line(wavelength_grid) + offsets[i], c='k', ls='-', lw=0.25)
        #ax.fill_between(wavelength_grid.value, scale[i]*pcygni_line(wavelength_grid) + offsets[i],
        #        scale[i]*fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
        #                    fc='#ebc3d4', alpha=0.5) #ebc3d4
        # place text stating X(Sr) at an appropriate height
        desired_text_loc = 4_000
        valid_flux = ~np.isnan(spec_ep[:, 1])  # Mask for valid flux values
        valid_indices = np.where(valid_flux)[0]  # Indices of valid flux
        closest_index = valid_indices[np.argmin(np.abs(spec_ep[valid_indices, 0] - desired_text_loc))]
        loc_flux = spec_ep[closest_index, 1]

        ax.text(x=desired_text_loc, y=offsets[i] + loc_flux - 0.1E-16, ha='left',
                      s='$X_{Sr}=' + f'{mass_fractions[i]*100:.3f} \%$', c=colors[i])
        ax.plot(wavelength_grid, scale[i]*fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
                       ls='-', color='dimgray', #label=f"{T:.2f} $\pm$ {T_sigma:.2f}",
                       lw=0.5,)
        nan_mask = ~np.isnan(spec_ep[:, 1])
        ax.step(spec_ep[nan_mask,0], scale[i]*spec_ep[nan_mask,2]+ offsets[i], ls='-', color='darkgray', alpha=0.4, lw=0.75) #/fitted_cont.eval(wavelength_grid=spec_ep[:,0]) - i 
        ax.step(spec_ep[nan_mask,0], scale[i]*spec_ep[nan_mask,1]+ offsets[i], ls='-', color=colors[i], lw=0.75,
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

import os
import pandas as pd
import NLTE.collision_rates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import astropy.constants as const

import main.atomic_utils as atomic
import NLTE.NLTE_model
import utils

from NLTE.NLTE_model import NLTESolver, States, Environment, RadiativeProcess, CollisionProcess, \
	RecombinationProcess, PhotoionizationProcess, HotElectronIonizationProcess
from pcygni_5_Spec_Rel import PcygniCalculator

from lmfit import Model
from scipy.interpolate import interp1d
from astropy.units import Quantity
from collections.abc import Iterable
from fractions import Fraction


h = const.h
c = const.c
kB = const.k_B

# truncate all data above this level (for now)
MAX_ENERGY_LEVEL = 25_000.0 # cm-1, corresponds to 400 nm from ground state.
# other than Sr II
ionization_stages_names = ['Sr I', 'Sr III', 'Sr IV', 'Sr V']

# strontium mass fraction (not just Sr II, all stages)
mass_fraction = 0.0002#E3


xshooter_dir = './Spectral Series of AT2017gfo/1.43-9.4 - X-shooter/dereddened+deredshifted_spectra/'
file_idx = lambda day: [str(day) in name for name in os.listdir(xshooter_dir)].index(1)
file_names = os.listdir(xshooter_dir)
# call xshooter_data(1.43) or xshooter_data(1.4) to get the spectrum of t=1.43 days
xshooter_data = lambda day: np.loadtxt(xshooter_dir + file_names[file_idx(day)])
# NOTE: This assumes there's only one file with e.g. '+1.43d' in its filename in the dir

blackbody_t = {
    # t_d: T_obs, T_emit (Lorentz boost corrected)
    1.43: (5400, 4400),
	4.40: (3200, 2800)
}

def get_names(levels_df: pd.DataFrame) -> list:
	configs = levels_df['Configuration'].apply(lambda s: s.split('.')[1])
	return (configs + ' ' +  levels_df['Term'] + ' ' + levels_df['J']).tolist()


def get_LTE_pops(energies: np.array, electron_temperature: float) -> np.array:
    beta = 1/(const.k_B * electron_temperature *u.K)
    n_LTE = np.exp(-beta*energies)
    return n_LTE/np.sum(n_LTE)


def blackbody_flux(wav0, T, amplitude, z, disable_units=False):
	wav = wav0*(1+z)
	flux = amplitude * 2*h*c**2/(wav**5) * (1 / (np.exp(h*c/(wav*kB*T)) - 1))#)
	if disable_units == False:
		flux = flux.to('erg s-1 cm-2 AA-1')
	return flux


def pcygni_interp(wav, v_out, v_phot, tau, resonance_wav, v1=0.22, ve=0.2, t_0=(1.43*u.day).to('s')):
	wav_grid, pcygni_profile = PcygniCalculator(t=t_0, vmax=v_out * const.c,
								 vphot=v_phot * const.c, tauref=tau, vref=v1 *
								 const.c, ve=ve * const.c,
								 lam0=resonance_wav).calc_profile_Flam(npoints=100)
	# the PCygni calculator evaluated the profile at certain points it decided
	# we need to interpolate between these to obtain values we want to plot the profile at
	interpolator = interp1d(wav_grid, pcygni_profile, bounds_error=False, fill_value=1)
	return interpolator(wav)

def blackbody_with_pycygnis(wavelength_grid, T, taus, line_wavelengths, v_out, v_phot, flux_amp,
									ve=0.2, occul=1., redshift=0., display=False):
	# if a list or np.array is passed instead of one line
	pcygni_profiles = []
	if isinstance(taus, Iterable): 
		for tau, line_res in zip(taus, line_wavelengths):
			line_adjust = pcygni_interp(wavelength_grid, v_out, v_phot, tau, line_res,
                               ve=ve)
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
		h, l = zip(*sorted(zip(handles, labels),
					# sort the legend display
            		key = lambda x: x[1]))
		ax2.legend(h, l, loc='lower right')
		plt.show()
	# rescale the flux to match the blackbody continuum and apply the combined PCygni line profiles
	return product_profiles*blackbody_flux(wavelength_grid, T=T, amplitude=flux_amp, z=0.)


def compute_nlte_opacities(T_electron, T_obs, mass_fraction, t_d, line_wavelengths):
	environment = Environment(T_phot=T_electron, 
					  photosphere_velocity=0.25,
					  line_velocity=0.3,
					  t_d=t_d)
	solver = NLTESolver(environment=environment,
				states=SrII_states,
				processes=[RadiativeProcess(SrII_states, environment)]
				)
	t, n = solver.solve(1e6)
	tau_matrix = NLTE.NLTE_model.pop_to_tau(environment, SrII_states, n[:,-2:-1],
	                                            A=solver.processes[0].A,
	                                            mass_fraction=mass_fraction)
	optical_depths = []
	line_wavelengths = []
	# Now, for each line's opacity
	for i in range(tau_matrix.shape[0]):
		for j in range(tau_matrix.shape[0]):
			if i <= j: continue
			optical_depths.append(tau_final[i,j])
			line_wavelengths.append(np.abs(SrII_states.energies[i] - SrII_states.energies[j]) \
   						.to('AA',equivalencies=u.spectral()))
	return optical_depths


'''def fit_blackbody(wavelength, observed_flux, masked_regions):
	planck_model = Model(blackbody_flux)
	pars = planck_model.make_params(
						T=5400,
						amplitude=np.max(observed_flux)*1E-6, # takes care of units, etc.
						z=0.,
						disable_units=True,
					)
	mask = np.ones_like(wavelength, dtype=bool)
	for left, right in masked_regions:
		mask &= ~((wavelength > left) & (wavelength < right))
	pars['z'].set(vary=False) # this should be kept constant
	pars['disable_units'].set(vary=False)
	fit = planck_model.fit(observed_flux, pars, wav0=wavelength[mask])
	print('Blackbody Fit Report\n ------\n', fit.fit_report())
	return fit'''

# NOTE: Loading of the line data happens in NLTE.NLTE_model.py, this is just for states
def get_strontium_states():
	# Energy levels from NIST
	levels = atomic.loadEnergyLevelsNIST(file_path='atomic_data/SrII_levels_NIST.csv')
	# there's an odd-parity state of the 4p6.8p J=1/2 whose energy is not listed in NIST. This has been discarded
	# it shouldn't matter as it's around ~75,000 cm-1 anyways

	# the ionization limit is included in this, so remove that and keep it separate	
	ionization_limit = levels[levels['Configuration'].str.contains('Sr III')]
	levels.drop(ionization_limit.index, inplace=True)

	# drop all energy levels above a certain energy threshold for now
	levels = levels[levels['Level (cm-1)'] < MAX_ENERGY_LEVEL]

	#state_names = SrII_levels['Configuration'] 
	level_energies = levels['Level (cm-1)'].to_numpy() / u.cm
	states_instance = States(names=get_names(levels),
					multiplicities=levels['g'].to_numpy(),
					energies=level_energies.to(u.eV, equivalencies=u.spectral()),
					# for now, I ignore Sr III and only load Sr II levels
					ionization_species=ionization_stages_names,
				)
	return states_instance

# deprecated
def load_strontium_line_data():
	# A-values for radiative line transitions
	SrII_lines_NIST_all = atomic.loadRadiativeTransitionsNIST('./atomic_data/SrII_lines_NIST_all.csv',
												sep=',')

	SrII_lines_NIST = SrII_lines_NIST_all.dropna(subset=['Aki(s^-1)'])
	# drop lines for all upper levels above max_energy_level
	SrII_lines_NIST = SrII_lines_NIST[SrII_lines_NIST['Ek(cm-1)'].astype(float) < MAX_ENERGY_LEVEL]



if __name__ == "__main__":
	SrII_states = get_strontium_states()
	SrII_states.texify_names() # TODO: make this get called automatically post-init in NLTE_model.py

	environment = Environment(T_phot=4400, 
					  photosphere_velocity=0.25,
					  line_velocity=0.3,
					  t_d=1.43)

	solver = NLTESolver(environment=environment,
					states=SrII_states,
					processes=[RadiativeProcess(SrII_states, environment),
							CollisionProcess(SrII_states, environment),
							HotElectronIonizationProcess(SrII_states, environment),
							RecombinationProcess(SrII_states, environment),
							#PhotoionizationProcess(SrII_states, environment)
					])
	 
	sr_coll_matr = CollisionProcess(SrII_states, environment).get_transition_rate_matrix()
	recombination_matrix = RecombinationProcess(states=SrII_states, environment=environment).get_transition_rate_matrix()
	nonthermal_matrix = HotElectronIonizationProcess(SrII_states, environment).get_transition_rate_matrix()

	utils.display_rate_timescale(recombination_matrix, SrII_states.tex_names + ionization_stages_names, 'Recombination')
	utils.display_rate_timescale(nonthermal_matrix, SrII_states.tex_names + ionization_stages_names, 'Non-thermal Ionization')
	
	"""
	Testing something
	"""
	t, n = solver.solve(1e6)
	'''tau, n, nlte_solver = NLTE.NLTE_model.solve_NLTE_sob(environment,
										states=SrII_states,
										processes=solver.processes, #n[:, i:i+1],
										mass_fraction=mass_fraction)'''
	# TODO: instead of taking solver.processes[0].A give a better way of returning the A matrix
	taus = np.array([NLTE.NLTE_model.pop_to_tau(environment, SrII_states, n[:,i:i+1],
											 solver.processes[0].A, mass_fraction) for i in range(n.shape[1])])
	# to pick out the steady state of the differential equation solution, just get the last item
	tau_final = taus[-1,:,:]
	optical_depths = []
	line_wavelengths = []

	# Now, for each line's opacity
	for i in range(tau_final.shape[0]):
		for j in range(tau_final.shape[0]):
			if i <= j: continue
			optical_depths.append(tau_final[i,j])
			line_wavelengths.append(np.abs(SrII_states.energies[i] - SrII_states.energies[j]) \
   						.to('AA',equivalencies=u.spectral()))

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
	

	fig, ax = plt.subplots(figsize=(8,6))

	# put the telluric masks
	flux_min_grid = -3.5E-16 * np.ones(100)
	flux_max_grid = 3E-16 * np.ones(100)
	for (left, right) in telluric_cutouts_albert:
		horizontal_grid = np.linspace(left, right, 100)
		ax.fill_between(horizontal_grid, flux_max_grid, flux_min_grid, fc='lightgray')
  
	# now plot the spectra, blackbody + pcygni fits
	wavelength_grid = np.linspace(2000, 23000, 10_000) * u.AA

	#blackbody_fit = fit_blackbody(spectrum_dat[:,0], spectrum_dat[:,1], masked_regions=all_masked_regions)
	amplitude = 1.33E-22

	epochs = [1.43, 2.42, 3.41, 4.4]
	offsets = np.array([0., -1.5, -2.5, -3.5])*1E-16
	for i, epoch in enumerate(epochs):
		colors = mpl.colormaps['Spectral_r'](np.linspace(0, 1.0, len(epochs)))
		spec_ep = xshooter_data(day=epoch)
		# sets the amplitude, etc. of the fitted blackbody
		ep1_fitted_cont = utils.fit_blackbody(spec_ep[:,0], spec_ep[:,1], all_masked_regions)
		# only thing that needs to be tuned is then the mass_fraction
		#ep1_fitted_line = utils.fit_planck_with_pcygni()
		T = ep1_fitted_cont.params['T'].value
		T_sigma = ep1_fitted_cont.params['T'].stderr
		ax.plot(wavelength_grid, ep1_fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
		  			 ls='--', color='dimgray', #label=f"{T:.2f} $\pm$ {T_sigma:.2f}",
					   lw=1.,)
		ax.plot(spec_ep[:,0], spec_ep[:,2] + offsets[i], ls='-', color='darkgray', lw=0.75)
		ax.plot(spec_ep[:,0], spec_ep[:,1] + offsets[i], ls='-', color=colors[i], lw=0.75,
		  				label=f'$t={epoch}$ days')
	ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
	ax.set_xlabel("Wavelength [$\mathrm{\AA}$]")

	ax.legend(loc='upper right')
	#ax.set_xscale('log')
	plt.tight_layout()
	plt.savefig('fitted_cpygni.png', dpi=300)
	plt.show()
	
	# plot how the optical depth and populations achieve, and the ionization balance
	
	fig, axes = plt.subplots(1, 3, figsize=(15,6))
	print("SHAPE OF TAU: ", taus.shape)
	print("Len of optical depths", len(optical_depths))
	for i in range(len(SrII_states.names)):
		for j in range(len(SrII_states.names)): 
			if i <= j: continue
			axes[0].plot(t, taus[:,i,j], label=SrII_states.tex_names[i] + r'$\to$ ' + SrII_states.tex_names[j])
	axes[0].set_ylabel(r'Sobolev Optical Depth [$\tau$]')
 	
	# also plot the nLTE populations
	level_colors = mpl.colormaps['plasma'](np.linspace(0, 1, len(SrII_states.names)))
	# and for comparison, LTE populations
	LTE_pops = get_LTE_pops(SrII_states.energies, environment.T_electrons)
	for i in range(len(SrII_states.names)):
		axes[1].plot(t, n[i], label=SrII_states.tex_names[i], c=level_colors[i])
		axes[1].axhline(y=LTE_pops[i], xmin=0, xmax=1, ls='--', c=level_colors[i])
	axes[1].set_ylabel('Level Occupancy')
	

	#LTE_opacities = NLTE.NLTE_model.pop_to_tau(np.array([LTE_pops]))
	srII_ion_stages = np.append(
								[n[:len(SrII_states.names), :].sum(axis=0)], # sum individual Sr II levels for total Sr II
								n[len(SrII_states.names):, :], # remaining Sr I, III, IV, ..
						 axis=0) / (environment.n_He)
	ion_stage_names = ['Sr II'] + ionization_stages_names
	axes[2].set_ylabel("Ionization Fraction")
	axes[2].set_yscale('log')
	print("shape of this thing*****")
	print(srII_ion_stages.shape)
	print("*****------")
	print(srII_ion_stages[:,-1])
	print("sum of ionization stages:", srII_ion_stages[:,-1].sum())
	ion_stage_colors = mpl.colormaps['plasma'](np.linspace(0, 1, len(ion_stage_names)+1))
	for i, (ion, name) in enumerate(zip(srII_ion_stages, ion_stage_names)):
		axes[2].plot(t, srII_ion_stages[i], label=name, c=ion_stage_colors[i])
	for ax in axes:
		ax.legend(loc='lower right')
		ax.set_xlabel("Time")
		ax.set_yscale("log")
		ax.set_xscale("log")
	plt.tight_layout()
	plt.show()

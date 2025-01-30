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

from functools import partial
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
mass_fraction = 0.0002 # for initialization, will be fitted later


xshooter_dir = './Spectral Series of AT2017gfo/1.43-9.4 - X-shooter/dereddened+deredshifted_spectra/'
file_idx = lambda day: [str(day) in name for name in os.listdir(xshooter_dir)].index(1)
file_names = os.listdir(xshooter_dir)
# call xshooter_data(1.43) or xshooter_data(1.4) to get the spectrum of t=1.43 days
xshooter_data = lambda day: np.loadtxt(xshooter_dir + file_names[file_idx(day)])
# NOTE: This assumes there's only one file with e.g. '+1.43d' in its filename in the dir

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


def pcygni_interp(wav_to_interp_at, v_out, v_phot, tau, resonance_wav, vref=0.22, ve=0.2, t_0=(1.43*u.day).to('s')):
	wav_grid, pcygni_profile = PcygniCalculator(t=t_0, vmax=v_out * const.c,
								 vphot=v_phot * const.c, tauref=tau, vref=vref *
								 const.c, ve=ve * const.c,
								 lam0=resonance_wav).calc_profile_Flam(npoints=100)
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


def get_line_depths_wavelengths(tau_matrix, states):
	"""
	Takes the N x N tau matrix, and gives a list of optical depths
		(and corresponding line wavelengths)
	"""
	optical_depths = []
	line_wavelengths = []
	#print("SHAPE OF THE TAU MATRIX", tau_matrix.shape)
	for i in range(tau_matrix.shape[0]):
		for j in range(tau_matrix.shape[0]):
			if i <= j: continue
			optical_depths.append(tau_matrix[i,j])
			line_wavelengths.append(np.abs(states.energies[i] - states.energies[j]) \
										.to('AA',equivalencies=u.spectral()))
	return optical_depths, line_wavelengths


def composite_pcygni(wavelength_grid, mass_fraction, v_out, v_phot, vref, ve,
					 	 environment, states, level_occupancy, A_rates):
	"""
	Given a tau matrix, evaluate a line
	NOTE: So far this doesn't allow the 400nm and 1um features to trace different velocity fields
	TODO: Taking the product is a slight simplification, treat that more properly.
	"""
	line_profiles = [] 
	tau_matrix = NLTE.NLTE_model.pop_to_tau(environment, states, level_occupancy, A_rates, mass_fraction)
	optical_depths, wavelengths = get_line_depths_wavelengths(tau_matrix, states)
	for tau, resonance_wav in zip(optical_depths, wavelengths):
		line_profiles.append(pcygni_interp(wavelength_grid, v_out=v_out, v_phot=v_phot, tau=tau,
					 resonance_wav=resonance_wav, vref=vref, ve=ve, t_0=(epoch * u.day).to('s')))
	product = np.prod(np.array(line_profiles), axis=0)
	if np.any(np.isnan(product)) > 0:
		print("THE PRODUCT CONTAINS NANs")
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
	print("THIS WAS CROSSED!")
	return blackbody_continuum.eval(wavelength_grid=wavelength_grid)*SpectrumModel.fit(normed_spec[:,1], params=params,
		 wavelength_grid=normed_spec[:,0], method='differential_evolution')

def init_tau_solve(states, environment, processes, mass_fraction):
	solver = NLTESolver(environment, states, processes=processes)
	t, level_occupancy_t = solver.solve(1e6)
	# NOTE: One doesn't have to compute $\tau$ at each time step. This was just to see how it evolves.
	tau_matrices = [
		NLTE.NLTE_model.pop_to_tau(environment,
			states,
			level_occupancy_t[:, i:i+1],
			get_process(solver, RadiativeProcess).A,
			mass_fraction,
			)
		# for each time step
		for i in range(len(t))
	]
	return t, level_occupancy_t, tau_matrices


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

	print("THIS WAS CALLED!")
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


def compute_nlte_opacities(T_electron, mass_fraction, t_d, line_wavelengths):
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


def display_time_evolution(t, level_occupancies, taus):
	# plot how the optical depth and populations achieve, and the ionization balance
	
	fig2, axes = plt.subplots(1, 3, figsize=(15,6))
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
		axes[1].plot(t, level_occupancies[i], label=SrII_states.tex_names[i], c=level_colors[i])
		axes[1].axhline(y=LTE_pops[i], xmin=0, xmax=1, ls='--', c=level_colors[i])
	axes[1].axhline(y=1, xmin=0, xmax=1, c='k', ls='-')
	axes[1].set_ylabel('Level Occupancy')
	

	#LTE_opacities = NLTE.NLTE_model.pop_to_tau(np.array([LTE_pops]))
	srII_ion_stages = np.append(
								[level_occupancies[:len(SrII_states.names), :].sum(axis=0)], # sum individual Sr II levels for total Sr II
								level_occupancies[len(SrII_states.names):, :], # remaining Sr I, III, IV, ..
						 axis=0)
	ion_stage_names = ['Sr II'] + ionization_stages_names
	axes[2].set_ylabel("Ionization Fraction")
	axes[2].set_yscale('log')
	ion_stage_colors = mpl.colormaps['plasma'](np.linspace(0, 1, len(ion_stage_names)+1))
	for i, (ion, name) in enumerate(zip(srII_ion_stages, ion_stage_names)):
		axes[2].plot(t, ion, label=name + f' ({100*ion[-1]:.1f}\%)', c=ion_stage_colors[i])
	for ax in axes:
		ax.legend(loc='lower right')
		ax.set_xlabel("Time")
		ax.set_yscale("log")
		ax.set_xscale("log")
	plt.tight_layout()
	plt.show()


absorption_region = (7000, 10500)
if __name__ == "__main__":
	SrII_states = get_strontium_states()
	SrII_states.texify_names() # TODO: make this get called automatically post-init in NLTE_model.py

	fig, ax = plt.subplots(figsize=(8,6))

	# put the telluric masks
	flux_min_grid = -3.5E-16 * np.ones(100)
	flux_max_grid = 3E-16 * np.ones(100)
	for (left, right) in telluric_cutouts_albert:
		horizontal_grid = np.linspace(left, right, 100)
		ax.fill_between(horizontal_grid, flux_max_grid, flux_min_grid, fc='lightgray')
  
	# now plot the spectra, blackbody + pcygni fits
	wavelength_grid = np.linspace(2500, 23000, 10_000) * u.AA

	T_elec_epochs = {1.43: 4400,
					2.42: 3200,
					3.41: 2900,
					4.40: 2800}
	
	offsets = np.array([0., -1.5, -2.5, -3.5])*1E-16

	v_outs = [0.42, 0.35, 0.28, 0.25]
	v_phots = [0.2, 0.13, 0.12, 0.11]
	ve_s = [0.6] * 4 # ve and v_ref aren't going to change the required mass fraction
	v_refs = [0.22] * 4
	mass_fractions = [0.0001, 0.0001, 0.0001, 0.0001]

	for i, (epoch, T_e) in enumerate(T_elec_epochs.items()):
		print("COUNTING!: ", i, epoch, T_e)
		colors = mpl.colormaps['Spectral_r'](np.linspace(0, 1.0, len(T_elec_epochs)))
		spec_ep = xshooter_data(day=epoch)
		# sets the amplitude, etc. of the fitted blackbody
		fitted_cont = utils.fit_blackbody(spec_ep[:,0], spec_ep[:,1], all_masked_regions)
		# only thing that needs to be tuned is then the mass_fraction
		#fitted_spectral_line = utils.fit_planck_with_pcygni(spec_ep[:,0], spec_ep[:,1], telluric_cutouts_albert)
		T = fitted_cont.params['T'].value
		T_sigma = fitted_cont.params['T'].stderr
		
		environment = Environment(t_d=epoch,
								T_phot=T_e,
								mass_fraction=mass_fractions[i],
								atomic_mass=88,
								photosphere_velocity=v_phots[i],
								line_velocity=v_phots[i],
								T_electrons=T_e)
		solver = NLTESolver(environment, SrII_states, processes=
					  			[RadiativeProcess(SrII_states, environment),
								CollisionProcess(SrII_states, environment),
								HotElectronIonizationProcess(SrII_states, environment),
								RecombinationProcess(SrII_states, environment),
								#PhotoionizationProcess(SrII_states, environment)
						])
		print(f"Given mass fraction X={mass_fractions[i]}, n_Sr={environment.n_He} at t={epoch}")
		#sr_coll_matr = get_rate_matrix(solver, CollisionProcess)
		#recombination_matrix = get_rate_matrix(solver, RecombinationProcess)
		#nonthermal_matrix = get_rate_matrix(solver, HotElectronIonizationProcess)
		#utils.display_rate_timescale(recombination_matrix, SrII_states.tex_names + ionization_stages_names, 'Recombination')
		#utils.display_rate_timescale(nonthermal_matrix, SrII_states.tex_names + ionization_stages_names, 'Non-thermal Ionization')

		# estimate non-LTE atomic populations
		t, level_occupancy = solver.solve(1e6)
		# use these populations and fit a spectrum
		'''fitted_spec = fit_spectrum(spec_ep, environment, SrII_states, solver,
							 		 level_occupancy[:, -2:-1], # steady state level occupancy
									 fitted_cont,
									 absorption_region) # blackbody parameters
		x_sr = fitted_spec.pars['mass_fraction']#.value and .stderr '''
		tau_matrices = np.array([
			NLTE.NLTE_model.pop_to_tau(environment,
				SrII_states,
				level_occupancy[:, j:j+1],
				get_process(solver, RadiativeProcess).A,
				mass_fractions[i],
				)
			# for each time stepß
			for j in range(len(t))
		])
		line_depths, resonance_wavelengths = get_line_depths_wavelengths(tau_matrices[-1], SrII_states)
		pcygni_line = lambda wav: blackbody_with_pcygnis(wav, line_depths, resonance_wavelengths,
									  fitted_cont, t_0=(epoch * u.day).to('s'), v_out=v_outs[i], v_phot=v_phots[i], display=False)
							   #environment=environment, states=SrII_states, level_occupancy=level_occupancy[:, -2:-1], A_rates=)
		ax.plot(wavelength_grid, pcygni_line(wavelength_grid) + offsets[i], c='k', ls='-', lw=0.25)
		ax.fill_between(wavelength_grid.value, pcygni_line(wavelength_grid) + offsets[i],
				   fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
				  			fc='#ebc3d4', alpha=0.7)
		#ax.fill_between(spec_ep[:,0], my_choice(spec_ep[:,0]) + offsets[i],spec_ep[:,1] + offsets[i],color='lightpink')
		#t, n, tau, fitted_spec = fit_pcygnis(spec_ep, sr_solver, environment, epoch=epoch, fitted_planck=fitted_cont,
		#					wavelength_grid=wavelength_grid, absorption_region=absorption_region)

		ax.plot(wavelength_grid, fitted_cont.eval(wavelength_grid=wavelength_grid) + offsets[i],
		  			 ls='-', color='dimgray', #label=f"{T:.2f} $\pm$ {T_sigma:.2f}",
					   lw=0.5,)
		ax.plot(spec_ep[:,0], spec_ep[:,2]+ offsets[i], ls='-', color='darkgray', lw=0.75) #/fitted_cont.eval(wavelength_grid=spec_ep[:,0]) - i 
		ax.plot(spec_ep[:,0], spec_ep[:,1]+ offsets[i], ls='-', color=colors[i], lw=0.75,
		  				label=f'$t={epoch}$ days')
		#plt.show()
		#display_time_evolution(t, environment, level_occupancy, tau_matrices)
	ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
	ax.set_xlabel("Wavelength [$\mathrm{\AA}$]")

	ax.legend(loc='upper right')
	#ax.set_xscale('log')
	plt.tight_layout()
	plt.savefig('fitted_cpygni.png', dpi=300)
	plt.show()

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

def pcygni_interp(wav, v_out, v_phot, tau, resonance_wav, v1=0.22, ve=0.2, t_0=(1.43 * u.day).to('s').value):
	wav_grid, pcygni_profile = PcygniCalculator(t=t_0 * u.s, vmax=v_out * const.c,
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
			optical_depths.append(tau[i,j])
			line_wavelengths.append(np.abs(SrII_states.energies[i] - SrII_states.energies[j]) \
   						.to('AA',equivalencies=u.spectral()))
	return optical_depths


def fit_blackbody(wavelength, observed_flux, masked_regions):
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
	return fit


blackbody_t = {
    # t_d: T_obs, T_emit
    1.43: (5400, 4400),
	4.40: (3200, 2800)
}

if __name__ == "__main__":
	# truncate all data above this level (for now)
	max_energy_level = 25_000.0 # cm-1

	# Energy levels from NIST
	SrII_levels = atomic.loadEnergyLevelsNIST(file_path='atomic_data/SrII_levels_NIST.csv')
	
	# the ionization limit is included in this, so remove that and keep it separate
	ion_limit_cond = SrII_levels['Configuration'].str.contains('Sr III')
	
	SrII_ionization_limit = SrII_levels[ion_limit_cond]
	SrII_levels.drop(SrII_levels[ion_limit_cond].index, inplace=True)
	SrII_levels = SrII_levels[SrII_levels['Level (cm-1)'] < max_energy_level]

	state_names = SrII_levels['Configuration'] 
	level_energies = SrII_levels['Level (cm-1)'].to_numpy() / u.cm
	ionization_stages_names = ['Sr I', 'Sr III', 'Sr IV', 'Sr V']
	SrII_states = States(names=get_names(SrII_levels),
					multiplicities=SrII_levels['g'].to_numpy(),
					energies=level_energies.to(u.eV, equivalencies=u.spectral()),
					# for now, I ignore Sr III and only load Sr II levels
					ionization_species=ionization_stages_names,
				)
	SrII_states.texify_names()
	# there's an odd-parity state of the 4p6.8p J=1/2 whose energy is not listed in NIST. This has been discarded
	# it shouldn't matter as it's around ~75,000 cm-1 anyways

	# A-values for radiative line transitions
	SrII_lines_NIST_all = atomic.loadRadiativeTransitionsNIST('./atomic_data/SrII_lines_NIST_all.csv',
												sep=',')

	SrII_lines_NIST = SrII_lines_NIST_all.dropna(subset=['Aki(s^-1)'])
	# drop lines for all upper levels above max_energy_level
	SrII_lines_NIST = SrII_lines_NIST[SrII_lines_NIST['Ek(cm-1)'].astype(float) < max_energy_level]

 
	environment = Environment(T_phot=4400, 
					  photosphere_velocity=0.25,
					  line_velocity=0.3,
					  t_d=1.43)

	solver = NLTESolver(environment=environment,
					states=SrII_states,
					processes=[RadiativeProcess(SrII_states, environment),
							CollisionProcess(SrII_states, environment),
							#HotElectronIonizationProcess(SrII_states, environment),
							#RecombinationProcess(SrII_states, environment),
							#PhotoionizationProcess(SrII_states, environment)
					])
	
	#he_states = States()
	#he_solver = NLTESolver(states=he_states, environment=environment)
	#print('Rate matrix for helium collisions', he_solver.processes[0].get_transition_rate_matrix())
	sr_coll_matr = CollisionProcess(SrII_states, environment).get_transition_rate_matrix()
	#sr_matrix_all = solver.get_transition_rate_matrix()
	#print("Fractional Contribution for Sr II")
	print("----")
	#print(sr_coll_matr/sr_matrix_all)
	print('----')
	print("Rate matrix for thermal collisions in Sr\n", sr_coll_matr)
	print('-----')

	"""
	Testing something
	"""
	#print("Number of bound states assumed for Sr II", len(SrII_levels))
	#print("Dimension of transition-rate matrix", solver.get_transition_rate_matrix().shape)

	mass_fraction = 0.0002#E3
	t, n = solver.solve(1e6)
	'''tau, n, nlte_solver = NLTE.NLTE_model.solve_NLTE_sob(environment,
										states=SrII_states,
										processes=solver.processes, #n[:, i:i+1],
										mass_fraction=mass_fraction)'''
	taus = [NLTE.NLTE_model.pop_to_tau(environment, SrII_states, n[:,i:i+1],solver.processes[0].A,
							mass_fraction) 
				for i in range(n.shape[1])]
	taus = np.array(taus)
	tau = taus[-1,:,:]
	optical_depths = []
	line_wavelengths = []
	# Now, for each line's opacity
	for i in range(tau.shape[0]):
		for j in range(tau.shape[0]):
			if i <= j: continue
			optical_depths.append(tau[i,j])
			line_wavelengths.append(np.abs(SrII_states.energies[i] - SrII_states.energies[j]) \
   						.to('AA',equivalencies=u.spectral()))
	# just to look at the how the optical depth saturates with time step
	#print("tau matrix", tau)
	#print(optical_depths)
	#print([w.value for w in line_wavelengths])

	spectra_dir = './Spectral Series of AT2017gfo/1.43-9.4 - X-shooter/dereddened+deredshifted_spectra/'
	epoch_file = 'AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase+1.43d_deredz.dat'
	epoch_t2_42 = 'AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase+2.42d_deredz.dat'
	spectrum_dat = np.loadtxt(spectra_dir + epoch_file)
	spectrum_t2_42 = np.loadtxt(spectra_dir + epoch_t2_42)
	spectrum_t3_41 = np.loadtxt(spectra_dir \
     					+ 'AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase+3.41d_deredz.dat')
	spectrum_t4_40 = np.loadtxt(spectra_dir \
     					+ 'AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase+4.40d_deredz.dat')
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
								[3200, 4000]
							], axis=0)
	fig, ax = plt.subplots(figsize=(8,6))

	# put the telluric masks
	flux_min_grid = -3.5E-16 * np.ones(100)
	flux_max_grid = 4E-16 * np.ones(100)
	for (left, right) in telluric_cutouts_albert:
		horizontal_grid = np.linspace(left, right, 100)
		ax.fill_between(horizontal_grid, flux_max_grid, flux_min_grid, fc='lightgray')
  
	# now plot the spectra, blackbody + pcygni fits
	wavelength_grid = np.linspace(2000, 22500, 10_000) * u.AA

	#blackbody_fit = fit_blackbody(spectrum_dat[:,0], spectrum_dat[:,1], masked_regions=all_masked_regions)
	amplitude = 1.33E-22
	blackbody_continuum = blackbody_flux(wavelength_grid, T=5400 * u.K,
                                      amplitude=amplitude, z=0.)
	synthetic_flux = blackbody_with_pycygnis(wavelength_grid,
                                          taus=optical_depths,
                                          line_wavelengths=line_wavelengths,
                                          v_out=0.42, # in terms of c
                                          v_phot=0.20,
                                          flux_amp=amplitude,
                                          T = 5400 * u.K,
                                          display=False # just to show a preview of the line profiles
                                          )

	ax.fill_between(wavelength_grid.value, blackbody_continuum.value, synthetic_flux.value,
                 			fc='cornflowerblue', alpha=0.7,
                    		label='Sr II ($X_{\mathrm{Sr II}} = ' + f'{mass_fraction}$)')
	ax.plot(wavelength_grid, blackbody_continuum, ls='-', c='k', lw=1., label='5400 K')
	ax.plot(wavelength_grid, synthetic_flux, ls='--', c='darkred', lw=1.)
 
	ax.plot(spectrum_dat[:,0], spectrum_dat[:,2], c='dimgray', lw=0.5, label='without telluric correction')
	ax.plot(spectrum_dat[:,0], spectrum_dat[:,1], c='darkred', lw=0.5, label='telluric corrected')

	y_offset = 1E-16
 
	# t=2.42
	ax.plot(spectrum_t2_42[:,0], spectrum_t2_42[:,2] - 1.25*y_offset, c='dimgray', lw=0.5)
	ax.plot(spectrum_t2_42[:,0], spectrum_t2_42[:,1] - 1.25*y_offset, c='darkred', lw=0.5)

	tau_t2_42 = compute_nlte_opacities(T_electron=3200,
                                              T_obs=3940, mass_fraction=mass_fraction,
                                              t_d=2.42,
                                            line_wavelengths=line_wavelengths)
	fit_t2_42 = blackbody_with_pycygnis(wavelength_grid,
                                          taus=tau_t2_42,
                                          line_wavelengths=line_wavelengths,
                                          v_out=0.35, # in terms of c
                                          v_phot=0.15,
                                          flux_amp=1.92*amplitude,
                                          T = 3940 * u.K,
                                          display=False # just to show a preview of the line profiles
                                          )
	planck_t2_42 = blackbody_flux(wavelength_grid, T=3940 * u.K,
                                      amplitude=1.92*amplitude, z=0.)
	planck_t3_41 = blackbody_flux(wavelength_grid, T=3420 * u.K,
                                      amplitude=2.5*amplitude, z=0.)
	planck_t4_40 = blackbody_flux(wavelength_grid, T=3330 * u.K,
                                      amplitude=2.5*amplitude, z=0.)
	ax.plot(wavelength_grid, planck_t2_42.value - 1.25*y_offset, ls='-', c='k', lw=1.,)

	# t=3.41
	ax.plot(spectrum_t3_41[:,0], spectrum_t3_41[:,2] - 2.25*y_offset, c='dimgray', lw=0.5)
	ax.plot(spectrum_t3_41[:,0], spectrum_t3_41[:,1] - 2.25*y_offset, c='darkred', lw=0.5)
	tau_t3_41 = compute_nlte_opacities(T_electron=2900,
                                              T_obs=3420, mass_fraction=mass_fraction,
                                              t_d=3.41,
                                            line_wavelengths=line_wavelengths)
	fit_t3_41 = blackbody_with_pycygnis(wavelength_grid,
                                          taus=tau_t3_41,
                                          line_wavelengths=line_wavelengths,
                                          v_out=0.3, # in terms of c
                                          v_phot=0.14,
                                          flux_amp=2.5*amplitude,
                                          T = 3420 * u.K,
                                          display=False # just to show a preview of the line profiles
                                          )
	ax.plot(wavelength_grid, planck_t3_41.value - 2.25*y_offset, ls='-', c='k', lw=1.,)

	#t_d =4.40
	ax.plot(spectrum_t4_40[:,0], spectrum_t4_40[:,2] - 3.25*y_offset, c='dimgray', lw=0.5)
	ax.plot(spectrum_t4_40[:,0], spectrum_t4_40[:,1] - 3.25*y_offset, c='darkred', lw=0.5)

	tau_t4_40 = compute_nlte_opacities(T_electron=2800,
                                              T_obs=3330, mass_fraction=mass_fraction,
                                              t_d=4.40,
                                            line_wavelengths=line_wavelengths)
	fit_t4_40 = blackbody_with_pycygnis(wavelength_grid,
                                          taus=tau_t4_40,
                                          line_wavelengths=line_wavelengths,
                                          v_out=0.3, # in terms of c
                                          v_phot=0.11,
                                          flux_amp=2.5*amplitude,
                                          T = 3330 * u.K,
                                          display=False # just to show a preview of the line profiles
                                          )
	ax.plot(wavelength_grid, planck_t4_40.value - 3.25*y_offset, ls='-', c='k', lw=1.,)

	ax.fill_between(wavelength_grid.value, planck_t2_42.value - 1.25*y_offset, fit_t2_42.value - 1.25*y_offset,
                 			fc='cornflowerblue', alpha=0.7)
	ax.fill_between(wavelength_grid.value, planck_t3_41.value - 2.25*y_offset, fit_t3_41.value - 2.25*y_offset,
                 			fc='cornflowerblue', alpha=0.7)
	ax.fill_between(wavelength_grid.value, planck_t4_40.value - 3.25*y_offset, fit_t4_40.value - 3.25*y_offset,
                 			fc='cornflowerblue', alpha=0.7)
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
	srII_ion_stages = np.append([np.sum(n[:len(SrII_states.names),:], axis=0)],n[len(SrII_states.names):,:], axis=0) \
						/ (mass_fraction*environment.n_He)
	ion_stage_names = ['Sr II'] + ionization_stages_names
	axes[2].set_ylabel("Ionization Fraction")
	axes[2].set_yscale('log')
	print("shape of this thing*****")
	print(srII_ion_stages.shape)
	print("*****------")
	print(srII_ion_stages)
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

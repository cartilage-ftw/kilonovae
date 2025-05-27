import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
# for reading HTML
import lxml
# for regex expressions
import re 
import roman # Sr3+ -> Sr IV

# personal utils for matplotlib code
import utils.plot_utils

from functools import cache
from astropy.units.quantity import Quantity
from astropy.constants import k_B, h, c, m_e
from matplotlib.lines import Line2D
from fractions import Fraction
from dataclasses import dataclass


@dataclass
class EnergyLevel:
	energy: Quantity
	valence_configuration: str
	term: str
	full_configuration: str
	J: str # fraction string (e.g. "1/2")
	n: int
	orbital: 'str'

@dataclass
class RadiativeLine:
	level_init: EnergyLevel
	level_final: EnergyLevel
	A_ki: float # the Einstein coefficient for spontaneous emission


@dataclass
class Atom:
	species: str # e.g. 'Sr'
	ionization_state: int # e.g. +1
	levels: iter
	radiative_lines: iter

	def __str__(self):
		"""
		NOTE: This doesn't handle negatively charged ions. So careful while using it for H-
		"""
		return self.species + ' ' + roman.toRoman(self.ionization_state)


def add_multiplicity_column(line_data):
	"""I forgot to request multiplicities and one day when I wanted to use it,
	the web form wasn't working anymore. I thought g_i = 2*J_i + 1 is easy to calculate

	Args:
		line_data (Pandas Dataframe):
	"""
	# in the .csv the J values are in fraction strings, e.g. '3/2' not floats
	J_i_floats = np.array([float(Fraction(line_data.iloc[i]['J_i'])) for i in range(len(line_data))])
	J_k_floats = np.array([float(Fraction(line_data.iloc[i]['J_k'])) for i in range(len(line_data))])

	g_i = 2*J_i_floats + 1
	g_k = 2*J_k_floats + 1

	line_data['g_i'] = g_i.astype(int)
	line_data['g_k'] = g_k.astype(int)


def add_TeX_names(df, outer_only=True):
    # use a regex expression to split 2P* into '2', 'P', '*'
    outer_conf = df['Configuration'].apply(lambda x: x.split('.')[1])
    df[['Spin Multiplicity', 'L', "Parity"]] = df['Term'].str.extract("(\d+)(\w+)(\*)")
    df['Parity'] = df['Parity'].str.replace('*', 'o')
    
    df['TeX Name'] = [
        	f'{conf} ^{spin}{L}^{parity}_{{{j}}}'
            for (conf, spin, L, parity, j) in 
            	zip(outer_conf,
              		df['Spin Multiplicity'],
                	df['L'],
                 	df['Parity'],
                 	df['J'])
          ]

    return df['TeX Name']

@cache
def loadEnergyLevelsNIST(file_path: str, new_format=False):
	level_data = pd.read_csv(file_path, sep='\t')
	# NOTE: NaN values should be handled carefully; here I've only dropped based on energy
	level_data.dropna(subset=['Level (cm-1)'], inplace=True)
	
	#levels_cm = level_data['Level (cm-1)'] / u.cm
	loaded_levels = []
	level_multiplicities = []
	for i in range(len(level_data)):
		J = np.nan 
		level_ = level_data.iloc[i]
		if '---' not in level_['J']:
			config = level_['Configuration']
			if len(config.split('.')) > 1:
				valence_config = config.split('.')[1]
			else:
				valence_config = config
			n = re.split('[a-z]', valence_config)
			orbital = re.match('[a-z]', valence_config)
			J = float(Fraction(level_['J']))
			#print(f"For configuration {level_['Configuration']}, taking n={n}, orbital: {orbital}. valence_config:{valence_config}")
		else:
			config = valence_config = 'Ionized Continuum'
		energy_level = EnergyLevel(energy=level_['Level (cm-1)'] / u.cm,
								valence_configuration=valence_config,
								full_configuration=config.replace('.',''),
								term = level_['Term'],
								J = level_['J'],
								# take 4p6.5s
								n = n,
								orbital = orbital)
		level_multiplicities.append(2*J + 1)
		loaded_levels.append(energy_level)
	level_data['Multiplicities'] = level_multiplicities
	if new_format == True:
		return loaded_levels
	return level_data


def loadRadiativeTransitionsNIST(file_path: str, drop_without_Aki=False, sep='\t'):
	line_data = pd.read_csv(file_path, sep=sep)
	# drop line data with no A values
	if drop_without_Aki == True:
		line_data.dropna(subset=['Aki(s^-1)'], inplace=True)
	return line_data


def calc_boltzmann(E: Quantity, T: Quantity):
	if not isinstance(E, Quantity):
		print("Please provide energies with appropriate units (eV or 1/cm)")
	E = E.to('eV', equivalencies=u.spectral())
	beta = (1/(k_B*T)).to('1/eV', equivalencies=u.temperature_energy())
	return np.exp(-beta*E)


def loadHTMLTable(html_path):
	"""
	Mulholland+24 paper had HTML tables that I got from the MNRAS article
	"""
	# for some reason it returns a list with the exact same contents twice
	print("Reading:", html_path)
	return pd.read_html(html_path)[0]

def loadSafronovaLines(file_path):
	"""
		fd
	"""
	data = pd.read_csv(file_path, sep=',')
	# convert config into something consistent with NIST energy level conventions
 
	'''n_i, J_i_str = re.split('[a-z]',data['Initial'])
	orbital_i = re.match(['a-z'], data['Initial'])
	# the above 2 lines will take '6s1/2' and assign n_i = '6', orbital_i = 's'
	# and J_i_str = '1/2'

	n_k, J_k_str = re.split('[a-z]', data['Final'])
	orbital_k = re.match('[a-z]', data['Final'])'''
	return data
	

def partition_func(levels: pd.DataFrame, T: float):
	if not isinstance(T, Quantity):
		T = T * u.K
	boltzmann = calc_boltzmann(levels['Level (cm-1)'].to_numpy() / u.cm, T)
	return np.sum(levels['g'].to_numpy() * boltzmann)


def drawGrotrian(levels: pd.DataFrame, lines: pd.DataFrame, line_label: str,
				 	draw_levels=True, line_color='blue'):
	# levels
	bound_states = levels[levels['J'] != '---']
	ionization_limit = levels[levels['J'] == '---']['Level (cm-1)'] # energy is 5th column
	J_vals = np.array([float(Fraction(x)) for x in bound_states['J']])
 
	# lines
	photoion_lines = lines[lines['Ek(cm-1)'].str.startswith('[')]
	# remove the [ ] to make them
	photoion_lines['Ek(cm-1)'] = photoion_lines['Ek(cm-1)'].str.replace('[', '').str.replace(']', '')
	print("These lines are photoionizing: ", photoion_lines)
	# get rid of them from the first list.
	lines.drop(photoion_lines.index, inplace=True)
 
	J_i = np.array([float(Fraction(x)) for x in lines['J_i']])
	J_k = np.array([float(Fraction(x)) for x in lines['J_k']])
	fig, ax = plt.subplots(figsize=(6,6))
 
	if draw_levels == True:
		# draw the levels
		ax.hlines(y=bound_states['Level (cm-1)'], xmin=J_vals - 0.25, xmax= 0.25 + J_vals,
						color='dimgray', lw=0.75)
		ax.axhline(y=ionization_limit.iloc[0], xmin=0, xmax=1, ls='-', c='red', lw=0.5, label='Ionization limit')
		# for displaying J quantum number as x-axis
		J_min = np.min(J_vals)
		J_max = np.max(J_vals)
		J_grid = np.linspace(J_min, J_max, int(J_max - J_min + 1))
		ticks = [str(Fraction.from_float(j)) for j in J_grid]
	# draw the lines
	ax.plot([J_i, J_k], [lines['Ei(cm-1)'], lines['Ek(cm-1)']], ls='-', c=line_color, lw=0.5)
	# draw maybe photoionizing lines too
	Ji_ion = np.array([float(Fraction(x)) for x in photoion_lines['J_i']])
	Jk_ion = np.array([float(Fraction(x)) for x in photoion_lines['J_k']])
	ax.plot([Ji_ion, Jk_ion], [photoion_lines['Ei(cm-1)'], photoion_lines['Ek(cm-1)']])
	# manually create a handle to represent the different NIST lines it draws
	nist_line_handle = Line2D([0], [0], color=line_color) 
	ax.set_xticks(J_grid, ticks)
	ax.set_xlabel("J")
	ax.set_ylabel("Energy [cm$^{-1}$]")

	handles, labels= ax.get_legend_handles_labels()
	handles.extend([nist_line_handle])
	labels.extend([line_label])
	ax.legend(handles, labels, loc='lower right')
	plt.tight_layout()
	plt.show()
	
	
if __name__ == '__main__':
	SrII_levels = loadEnergyLevelsNIST('../atomic_data/SrII_levels_NIST.csv')
	SrII_levels_newformat = loadEnergyLevelsNIST('../atomic_data/SrII_levels_NIST.csv', new_format=True)
	# NIST lines with experimental A_ki
	#SrII_lines_NIST = loadRadiativeTransitionsNIST('../atomic_data/SrII_lines_NIST.txt')
	# lines that also don't have measured A_ki
	SrII_lines_NIST_all = loadRadiativeTransitionsNIST('../atomic_data/SrII_lines_NIST_all.txt')
	SrII_lines_NIST = SrII_lines_NIST_all.dropna(subset=['Aki(s^-1)'])
	print("All enumerated lines in NIST", len(SrII_lines_NIST_all))
	# for display: Just the lines that are lacking A_ki
	SrII_lines_lacking = SrII_lines_NIST_all.drop(SrII_lines_NIST.index)
	for j in SrII_lines_NIST['J_i']:
		print(j)
	print('Number of lines lacking data:', len(SrII_lines_lacking))
	M24_tables = {'SrII_A': 'SrII_A_values',
				  'YII_A': 'YII_A_values',
				  'SrII_e': 'SrII_e_collisions',
				  'YII_e': 'YII_e_collisions'}
	
	Sr_lines_Safronova = loadSafronovaLines('../atomic_data/SrIITransitionRates_Safronova.csv')
	Sr_lines_M24 = loadHTMLTable('../atomic_data/Mulholland+24/' + M24_tables['SrII_A'] + '.html')
	#print(Sr_lines_M24)
	print("****")
	print(Sr_lines_Safronova)
	print("*****")
	drawGrotrian(SrII_levels, SrII_lines_NIST, 'NIST Experimental')
	drawGrotrian(SrII_levels, SrII_lines_lacking, 'Unknown', line_color='gray', draw_levels=False)
	#print(computeLTEpop(SrII_levels, T=4_000*u.K))
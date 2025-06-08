from dataclasses import dataclass

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plot_utils


data_root = '../Spectral Series of AT2017gfo/'


@dataclass
class Observation:
    telescope_name: str
    time_since_merger: str
    data_file_path: str


def load_MagE_LDSS(file_path='0.48-0.53 - MagE & LDSS'):
    """
    Units are Angstroms for x-axis
    Flux in erg/s/cm^2/Angstrom
    """
    folder_path = data_root + file_path
    data = []
    for data_file in os.listdir(folder_path):
        dat = pd.read_csv(folder_path + '/' + data_file, sep='\s+', skiprows=7,
                            names = ['Wavelength', 'Flux'])
        print(f"Wavelength range in {data_file}: {np.min(dat['Wavelength'])} to {np.max(dat['Wavelength'])}")
        data.append(dat)
    return data # first one is LDSS, second is MagE


LDSS_spec, MagE_spec = load_MagE_LDSS()

"""
Wavelength range in SSS17a-LDSS3-20170817_cal.flm: 3725.6 to 10101.2
Wavelength range in SSS17a-MagE-20170817_cal.flm: 3615.0 to 6932.4
"""

fix, ax = plt.subplots(figsize=(6,6))

ax.plot(LDSS_spec['Wavelength'], LDSS_spec['Flux'], label='LDSS $t=0.48$d') # I might be wrong in which is 0.48
ax.plot(MagE_spec['Wavelength'], MagE_spec['Flux'].to_numpy() + 1.5E-15, label='MagE $t=0.53$d')
ax.legend()
ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"Flux [erg/s/cm$^{2}$/$\mathrm{\AA}$]")
plt.tight_layout()
plt.show()          
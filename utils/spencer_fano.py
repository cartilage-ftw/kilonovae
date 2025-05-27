import pynonthermal

# Kozma & Fransson 1992 Figure 2 - Pure-Oxygen Plasma
x_e = 1e-2
ions = [
    # (Z, ion_stage, number_density)
    (8, 1, 1.0 - x_e),
    (8, 2, x_e),
]

sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=2000, verbose=True)
for Z, ion_stage, n_ion in ions:
    sf.add_ionisation(Z, ion_stage, n_ion)
    sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion, temperature=6000)

sf.solve(depositionratedensity_ev=3.0e3)

sf.analyse_ntspectrum()

sf.plot_yspectrum()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from SrII_modelling import get_LTE_pops, compute_LTE_ionizations#, SrII_states
from NLTE.NLTE_model import get_density_profile

from scipy.integrate import quad

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

def display_time_solution(t, level_occupancies, tau_all_timesteps, states, environment):
    # plot how the optical depth and populations achieve, and the ionization balance
    fig2, axes = plt.subplots(1, 3, figsize=(15,6))
    line_wavelengths = []
    for i in range(len(states.names)):
        for j in range(len(states.names)): 
            if i <= j: continue
            wavelength = np.abs(states.energies[i] - states.energies[j]).to('AA', equivalencies=u.spectral())
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
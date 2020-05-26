from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys = system_data(
    m_e,
    m_h,
    eps_sol,
    T,
    size_d,
    0.0,
    0.0,
    0.0,
    0.0,
    eps_sol,
    ext_dist_l,
)


def cou_energy(LS, sys):
    return -(sys.c_hbarc * pi)**2 * 0.5 / (sys.m_e + sys.m_h) * LS**2 / (
        sys.size_Lx *
        sys.size_Ly)**2 + sys.c_aEM * sys.c_hbarc / sys.eps_mat / LS


print(sys.exc_bohr_radius_mat())
print(sys.get_E_n(0.5))

labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]
sizes_vec = [
    (29.32, 5.43),
    (26.11, 6.42),
    (25.4, 8.05),
    (13.74, 13.37),
]
hwhm_vec = [
    (3.3, 0.81),
    (3.34, 1.14),
    (2.9, 0.95),
    (2.17, 1.85),
]

sizes_vec = array(sizes_vec)
hwhm_vec = array(hwhm_vec)

LS_samples_vec = sqrt(sizes_vec[:, 0]**2 + sizes_vec[:, 1]**2)
LS_up_vec = sqrt((sizes_vec[:, 0] + hwhm_vec[:, 0] * 0.5)**2 +
                 (sizes_vec[:, 1] + hwhm_vec[:, 1] * 0.5)**2)
LS_low_vec = sqrt((sizes_vec[:, 0] - hwhm_vec[:, 0] * 0.5)**2 +
                  (sizes_vec[:, 1] - hwhm_vec[:, 1] * 0.5)**2)

N_LS = 1 << 8

LS_vec = linspace(15, 35, N_LS)

colors = [
    matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8])
    for h in linspace(0, 0.7, LS_samples_vec.size)
]

ax[0].axhline(
    y=0,
    linestyle='-',
    linewidth=0.9,
    color='k',
)

for i in range(LS_samples_vec.size):
    sys.size_Lx, sys.size_Ly = sizes_vec[i]

    ax[0].plot(
        LS_vec,
        cou_energy(LS_vec, sys) * 1e3,
        linewidth=1.3,
        color=colors[i],
        label=r'%s: $%.1f\times%.1f$ nm' %
        (labels_vec[i], *tuple(sizes_vec[i])),
    )

    ax[0].plot(
        [LS_samples_vec[i]],
        [cou_energy(LS_samples_vec[i], sys) * 1e3],
        linestyle='',
        marker='o',
        markeredgecolor=colors[i],
        markerfacecolor=(1, 1, 1, 1),
        markersize=8.0,
        markeredgewidth=1.8,
    )

ax[0].plot(
    LS_vec,
    sys.c_aEM * sys.c_hbarc / sys.eps_mat / LS_vec * 1e3,
    linestyle='--',
    linewidth=0.6,
    label='Coulomb',
)

ax[0].set_xlim(LS_vec[0], LS_vec[-1])
ax[0].set_xticks(arange(15, 36, 5))

ax[0].set_xlabel(r'$L_S$ (nm)')
ax[0].set_ylabel(r'$E$ (meV)')

lg = ax[0].legend(
    loc=0,
    prop={'size': 11},
)
lg.get_title().set_fontsize(11)

plt.tight_layout()

fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'cou_cont_e_A1',
    transparent=True,
)

plt.show()

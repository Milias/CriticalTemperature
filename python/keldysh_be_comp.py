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
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]


def size_d_eff(sys):
    return sys.size_d * sys.eps_mat / sys.eps_r


def eta_func(eps_mat, eps_sol):
    return 0.5 * log((eps_mat + eps_sol) / (eps_mat - eps_sol))


N_k = 1 << 9

size_d = 1.0  # nm
eps_sol = 2
m_e, m_h, T = 0.22, 0.41, 294  # K

mu_e = -1e2

be_min = -200e-3

N_eps = 64
N_d = 4

eps_vec = logspace(log10(eps_sol), log10(200.0), N_eps)
eps_hn_vec = logspace(log10(eps_sol), log10(1000.0), N_eps)

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
be_sol = time_func(
    plasmon_det_zero_ke,
    N_k,
    mu_e,
    sys_sol.get_mu_h(mu_e),
    sys_sol,
    be_min,
)

d_vec = array([0.2, 0.5, 1, 5]) * sys_sol.exc_bohr_radius()

print(exciton_be_cou(sys_sol))
print(be_sol)

sys_ke_vec = array([
    system_data(m_e, m_h, eps_sol, T, d, eps)
    for eps, d in itertools.product(eps_vec, d_vec)
]).reshape(N_eps, N_d)

sys_cou_vec = array([
    system_data(m_e, m_h, eps, T, d, eps)
    for eps, d in itertools.product(eps_vec, d_vec)
]).reshape(N_eps, N_d)

be_cou_vec = zeros_like(sys_cou_vec)
be_qcke_vec = zeros_like(sys_ke_vec)
be_ke_vec = zeros_like(sys_ke_vec)
be_hn_vec = zeros_like(sys_ke_vec)


def save_be_data():
    be_cou_vec[:] = array([
        time_func(
            plasmon_det_zero_ke,
            N_k,
            mu_e,
            sys.get_mu_h(mu_e),
            sys,
            be_min,
        ) for sys in sys_cou_vec[:, 0].flatten()
    ]).reshape((N_eps, 1))

    be_qcke_vec[:] = array([
        time_func(
            plasmon_det_zero_qcke,
            N_k,
            mu_e,
            sys.get_mu_h(mu_e),
            sys,
            be_min,
        ) for sys in sys_ke_vec.flatten()
    ]).reshape(N_eps, N_d)

    be_ke_vec[:] = array([
        time_func(
            plasmon_det_zero_ke,
            N_k,
            mu_e,
            sys.get_mu_h(mu_e),
            sys,
            be_min,
        ) for sys in sys_ke_vec.flatten()
    ]).reshape(N_eps, N_d)

    be_hn_vec[:] = array([
        time_func(
            plasmon_det_zero_hn,
            N_k,
            mu_e,
            sys.get_mu_h(mu_e),
            sys,
            be_min,
        ) for sys in sys_ke_vec.flatten()
    ]).reshape(N_eps, N_d)

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/keldysh/be_comp_%s' % file_id,
        [
            be_cou_vec.flatten(),
            be_qcke_vec.flatten(),
            be_ke_vec.flatten(),
            be_hn_vec.flatten()
        ],
        extra_data={
            'eps_vec': eps_vec.tolist(),
            'd_vec': d_vec.tolist(),
            'eps_sol': eps_sol,
            'size_d': size_d,
            'm_e': m_e,
            'm_h': m_h,
            'T': T,
            'mu_e': mu_e,
        },
    )

    return file_id


#file_id = 'X-p05-BmQIaSzIPEPmeEbg'
file_id = 'GB2wOKCDRZ2Z3mpKx4o-oQ'
#file_id = time_func(save_be_data)

data = load_data('extra/keldysh/be_comp_%s' % file_id, globals())

eps_vec, d_vec, N_eps, N_d = array(eps_vec), array(d_vec), len(eps_vec), len(
    d_vec)

be_cou_vec[:] = data[0].reshape(N_eps, N_d)
be_qcke_vec[:] = data[1].reshape(N_eps, N_d)
be_ke_vec[:] = data[2].reshape(N_eps, N_d)
be_hn_vec[:] = data[3].reshape(N_eps, N_d)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, d_vec.size)
]

for (nd, d), c in zip(enumerate(d_vec), colors):
    x_line = (d / sys_sol.exc_bohr_radius())

    ax[0].axvline(
        x=x_line,
        linestyle='--',
        color=c,
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.8,
    )

eps_full_vec = logspace(log10(eps_sol), log10(5e2), N_eps)
d = d_vec[0]
bohr_vec = eps_full_vec * sys_sol.c_hbarc / sys_sol.c_aEM / sys_sol.m_p
energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / eps_full_vec / d

x_vec = 1 / (d * bohr_vec * (eps_sol / d / eps_full_vec)**2)

ax[0].semilogx(
    x_vec,
    be_sol / energy_norm,
    color='m',
    linestyle='-',
    linewidth=0.6,
    #linestyle='--',
    #dashes=(9.5, 10.),
    #dash_capstyle='round',
    label=r'$\tilde E_B^{sol}$',
)

for (nd, d), c in zip(enumerate(d_vec), colors):
    bohr_vec = eps_vec * sys_sol.c_hbarc / sys_sol.c_aEM / sys_sol.m_p
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / eps_vec / d

    x_vec = 1 / (d * bohr_vec * (eps_sol / d / eps_vec)**2)
    if nd == 0:
        ax[0].semilogx(
            x_vec,
            be_hn_vec[:, nd] / energy_norm,
            color='m',
            linestyle='-',
            linewidth=1.8,
            label=r'$\tilde E_B^{\mathcal{H}N}$',
            zorder=10,
        )
    elif nd == N_d - 1:
        ax[0].semilogx(
            x_vec,
            be_hn_vec[:, nd] / energy_norm,
            color='m',
            linestyle='-',
            linewidth=1.8,
            zorder=10,
        )

for (nd, d), c in zip(enumerate(d_vec), colors):
    bohr_vec = eps_vec * sys_sol.c_hbarc / sys_sol.c_aEM / sys_sol.m_p
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / eps_vec / d

    x_vec = 1 / (d * bohr_vec * (eps_sol / d / eps_vec)**2)

    ax[0].semilogx(
        x_vec,
        be_ke_vec[:, nd] / energy_norm,
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.9,
    )
    """
    ax[0].semilogx(
        x_vec,
        be_cou_vec[:, nd] / energy_norm,
        color=c,
        linestyle='-',
        linewidth=0.6,
    )
    """

for (nd, d), c in zip(enumerate(d_vec), colors):
    bohr_vec = eps_vec * sys_sol.c_hbarc / sys_sol.c_aEM / sys_sol.m_p
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / eps_vec / d

    x_vec = 1 / (d * bohr_vec * (eps_sol / d / eps_vec)**2)

    ax[0].semilogx(
        [x_vec[0]],
        [be_qcke_vec[0, nd] / energy_norm[0]],
        color=c,
        marker='o',
    )

    ax[0].semilogx(
        [x_vec[0]],
        [be_ke_vec[0, nd] / energy_norm[0]],
        marker='o',
        markeredgecolor='m',
        markerfacecolor='#FFFFFF',
    )

    ax[0].semilogx(
        x_vec,
        be_qcke_vec[:, nd] / energy_norm,
        color=c,
        linestyle='-',
        linewidth=1.8,
        label=r'$d^* / a_0$: $%.1f$' % (d / sys_sol.exc_bohr_radius()),
    )

ax[0].legend(loc='upper right')

ax[0].set_xlim(
    4e-1 /
    (d_vec[1] * eps_vec[0] * sys_sol.c_hbarc / sys_sol.c_aEM / sys_sol.m_p *
     (eps_sol / d_vec[1] / eps_vec[0])**2),
    200,
)

ax[0].set_ylim(-5, 0)

ax[0].set_xlabel(r'$(d^* / d) \cdot (d^* / a_0)$')
ax[0].set_ylabel(r'$\tilde E_B$')

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'be_comp_v8')

plt.show()

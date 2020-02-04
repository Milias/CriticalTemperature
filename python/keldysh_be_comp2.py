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
N_d = 5

eps_vec = logspace(log10(eps_sol), log10(200.0), N_eps)

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
be_sol = time_func(
    plasmon_det_zero_ke,
    N_k,
    mu_e,
    sys_sol.get_mu_h(mu_e),
    sys_sol,
    be_min,
)


def solve_eps(eps_mat, be_exc, det_zero_func):
    sys = system_data(m_e, m_h, eps_mat, T, 1.37, eps_mat)
    return be_exc - time_func(
        det_zero_func,
        N_k,
        mu_e,
        sys.get_mu_h(mu_e),
        sys,
        be_min,
    )


"""
eps_mat_solved = root_scalar(
    solve_eps,
    bracket=(3, 20),
    method='brentq',
    args=(-193e-3, plasmon_det_zero_ke),
)
eps_mat_cou = eps_mat_solved.root

eps_mat_solved = root_scalar(
    solve_eps,
    bracket=(3, 20),
    method='brentq',
    args=(-193e-3, plasmon_det_zero_qcke),
)
eps_mat_qcke = eps_mat_solved.root
"""

eps_mat_cou = 6.389276286774717
eps_mat_qcke = 12.929731608105103

print(eps_mat_cou / eps_sol)
print(eps_mat_qcke / eps_sol)

print('Approximate a_0: %.2f nm' % system_data(
    m_e,
    m_h,
    eps_sol,
    T,
    1.37,
    eps_mat_qcke,
).exc_bohr_radius_mat())

d_vec = array([
    0.1,
    0.2,
    1,
    1.37 / sys_sol.exc_bohr_radius(),
    10,
]) * sys_sol.exc_bohr_radius()

print(1.37 / sys_sol.exc_bohr_radius())

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
        time_func(plasmon_det_zero_ke, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_cou_vec[:, 0].flatten()
    ]).reshape(N_eps, 1)

    be_qcke_vec[:] = array([
        time_func(plasmon_det_zero_qcke, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_ke_vec.flatten()
    ]).reshape(N_eps, N_d)

    be_ke_vec[:] = array([
        time_func(plasmon_det_zero_ke, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_ke_vec.flatten()
    ]).reshape(N_eps, N_d)

    be_hn_vec[:] = array([
        time_func(plasmon_det_zero_hn, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_ke_vec.flatten()
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


#file_id = 'GB2wOKCDRZ2Z3mpKx4o-oQ'
#file_id = 'aaT4BjkOSPKqUrAfDPhdAQ'
#file_id = 'zPx8SiFzQhS1qItviCS5uQ'
#file_id = 'U0gcAzwjQAutd9Ih9Rg9NQ'
#file_id = '9oCu__E_RxSIGnoWMw8HdQ'
file_id = 'wwaR50NQRn-e_D3u_TDveg'
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
"""
ax[0].axvline(
    x=1,
    linestyle='-',
    color='k',
    linewidth=0.7,
)
"""

for (nd, d), c in zip(enumerate(d_vec), colors):
    x_vec = eps_vec / eps_sol

    ax[0].semilogx(
        x_vec,
        be_ke_vec[:, nd],
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.9,
    )

ax[0].semilogx(
    eps_vec / eps_sol,
    be_cou_vec[:, 0],
    color='m',
    linestyle='-',
    linewidth=0.6,
    label=r'$\mathcal{E}^{C}(\epsilon)$',
)

for (nd, d), c in zip(enumerate(d_vec), colors):
    x_vec = eps_vec / eps_sol
    """
    ax[0].axvline(
        x=sys_sol.exc_bohr_radius() / d,
        linestyle='-',
        color=c,
        linewidth=0.5,
    )
    """

    ax[0].semilogx(
        [x_vec[0]],
        [be_qcke_vec[0, nd]],
        color=c,
        marker='o',
    )

    if d / sys_sol.exc_bohr_radius() < 1:
        ax[0].semilogx(
            x_vec,
            be_qcke_vec[:, nd],
            color=c,
            linestyle='-',
            linewidth=1.8,
            label=r'$d^* / a_0$: $%.1f$' % (d / sys_sol.exc_bohr_radius()),
        )

    elif nd == 3:
        ax[0].semilogx(
            x_vec,
            be_qcke_vec[:, nd],
            color=c,
            linestyle='-',
            linewidth=1.8,
            label=r'$d^* / a_0$: CdSe',
        )

    else:
        ax[0].semilogx(
            x_vec,
            be_qcke_vec[:, nd],
            color=c,
            linestyle='-',
            linewidth=1.8,
            label=r'$d^* / a_0$: $%d$' % (d / sys_sol.exc_bohr_radius()),
        )

ax[0].semilogx(
    [eps_vec[0] / eps_sol],
    [be_ke_vec[0, 0]],
    marker='o',
    markeredgecolor='m',
    markerfacecolor='#FFFFFF',
)

for (nd, d), c in zip(enumerate(d_vec), colors):
    x_vec = eps_vec / eps_sol
    ax[0].semilogx(
        x_vec,
        be_hn_vec[:, nd],
        color=c,
        linestyle='--',
        dashes=(0.8, 4.),
        dash_capstyle='round',
        linewidth=1.0,
    )
"""
ax[0].semilogx(
    [eps_mat_cou / eps_sol],
    [-193e-3],
    color='m',
    marker='.',
    #markeredgecolor='m',
    #markerfacecolor='#FFFFFF',
)
"""

ax[0].semilogx(
    [eps_mat_qcke / eps_sol],
    [-193e-3],
    marker='o',
    markeredgecolor=colors[3],
    markerfacecolor='#FFFFFF',
)

ax[0].legend(loc=0)

ax[0].set_xticks([1, 10, 100])
ax[0].set_xticklabels([r'$1$', r'$10$', r'$100$'])
ax[0].set_yticks([
    be_sol,
    -1.5,
    -1,
    -0.5,
    #-193e-3,
    0,
])
ax[0].set_yticklabels([
    r'$\mathcal{E}^{C}_{sol}$',
    '$-1.5$',
    '$-1.0$',
    '$-0.5$',
    #'$\mathcal{E}_X$',
    '$0$',
])

ax[0].set_xlim(0.97, eps_vec[-1] / eps_sol)
ax[0].set_ylim(1.05 * be_sol, 0)

ax[0].set_xlabel(r'$d^* / d$')
ax[0].set_ylabel(r'$\mathcal{E}$ (eV)')

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'be_comp_B2')

plt.show()

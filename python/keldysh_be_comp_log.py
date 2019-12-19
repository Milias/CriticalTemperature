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

N_k = 1 << 9

size_d = 1.0  # nm
eps_sol = 2
m_e, m_h, T = 0.22, 0.41, 294  # K

mu_e = -1e2

be_min = -200e-3

N_eps = 48
N_d = 4

eps_vec = logspace(log10(eps_sol), log10(2000.0), N_eps)

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
be_sol = time_func(
    plasmon_det_zero_ke,
    N_k,
    mu_e,
    sys_sol.get_mu_h(mu_e),
    sys_sol,
    be_min,
)

d_vec = logspace(log10(0.1), log10(6), N_d)

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


file_id = '4kCWrFM9SQWuF1GDnAoEqA'
#file_id = 'nwDgsFrKQLCftAUWrWwPBA'
#file_id = save_be_data()

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

print('Bohr radius: %f nm' % sys_sol.exc_bohr_radius())
"""
ax[0].axhline(
    y=-193e-3,
    linewidth=0.7,
    color='g',
)
"""

for (nd, d), c in zip(enumerate(d_vec), colors):
    #"""
    energy_norm = array([
        sys.c_aEM / sys.eps_mat * sys.c_hbarc / d for sys in sys_ke_vec[:, nd]
    ])
    #"""
    #energy_norm = 1
    ax[0].loglog(
        eps_vec / eps_sol * d / sys_sol.exc_bohr_radius(),
        -be_ke_vec[:, nd] / energy_norm,
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.0,
    )

    ax[0].loglog(
        eps_vec / eps_sol * d / sys_sol.exc_bohr_radius(),
        -be_hn_vec[:, nd] / energy_norm,
        color=c,
        linestyle='--',
        dashes=(0.2, 2.),
        dash_capstyle='round',
        linewidth=1.3,
    )

    ax[0].loglog(
        eps_vec / eps_sol * d / sys_sol.exc_bohr_radius(),
        -be_qcke_vec[:, nd] / energy_norm,
        color=c,
        linestyle='-',
        linewidth=1.8,
        label=r'$d / a_0$: $%.1f$' % (d / sys_sol.exc_bohr_radius()),
    )

ax[0].legend(loc=0)

#ax[0].set_ylim(-be_qcke_vec[-1, -1], -be_sol)
ax[0].set_xlim(
    eps_vec[0] / eps_sol,
    eps_vec[-1] / eps_sol,
)
"""
ax[0].set_yticks([be_sol, -1.5, -1, -0.5, 0])
ax[0].set_yticklabels([r'$E_{B,sol}$', '$-1.5$', '$-1.0$', '$-0.5$', '$0$'])
"""

ax[0].set_xticks([0.1, 1, 10, 100, 1000])
ax[0].set_xticklabels(['$10^{-1}$', '$1$', '$10$', '$10^2$', '$10^3$'])
"""
ax[0].set_xticks([1, 10, 100])
ax[0].set_xticklabels(['$1$', '$10$', '$10^2$'])
"""

ax[0].set_xlabel(r'$(\epsilon / \epsilon_{sol}) \cdot (d / a_0)$')
ax[0].set_ylabel(
    r'$E_B \cdot \left( \frac{e^2}{4 \pi \epsilon_0 \epsilon d} \right)^{-1}$')

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'be_comp_log_v1')

plt.show()

from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

N_k = 1 << 9

size_d = 1  # nm
eps_sol = 2
m_e, m_h, T = 0.22, 0.41, 294  # K

mu_e = -1e2

be_min = -200e-3

N_eps = 64
N_ns = 5

eps_vec = logspace(log10(eps_sol), log10(20.0), N_eps)

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
be_sol = time_func(
    plasmon_det_zero_ke,
    N_k,
    mu_e,
    sys_sol.get_mu_h(mu_e),
    sys_sol,
    be_min,
)

print(exciton_be_cou(sys_sol))
print(be_sol)

size_d = sys_sol.exc_bohr_radius()

sys_ke_vec = array(
    [system_data(m_e, m_h, eps_sol, T, size_d, eps) for eps in eps_vec])

sys_cou_vec = array(
    [system_data(m_e, m_h, eps, T, size_d, eps) for eps in eps_vec])

be_cou_vec = zeros((N_eps, N_ns))
be_qcke_vec = zeros((N_eps, N_ns))
be_ke_vec = zeros((N_eps, N_ns))
be_hn_vec = zeros((N_eps, N_ns))


def save_be_data():
    be_cou_vec[:] = array([
        time_func(plasmon_det_zero_ke_ns, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_cou_vec
    ]).reshape(N_eps, N_ns)

    be_qcke_vec[:] = array([
        time_func(plasmon_det_zero_qcke_ns, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_ke_vec
    ]).reshape(N_eps, N_ns)

    be_ke_vec[:] = array([
        time_func(plasmon_det_zero_ke_ns, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_ke_vec
    ]).reshape(N_eps, N_ns)

    be_hn_vec[:] = array([
        time_func(plasmon_det_zero_hn_ns, N_k, mu_e, sys.get_mu_h(mu_e), sys,
                  be_min) for sys in sys_ke_vec
    ]).reshape(N_eps, N_ns)

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/keldysh/be_ns_%s' % file_id,
        [
            be_cou_vec.flatten(),
            be_qcke_vec.flatten(),
            be_ke_vec.flatten(),
            be_hn_vec.flatten()
        ],
        extra_data={
            'eps_vec': eps_vec.tolist(),
            'N_ns': N_ns,
            'eps_sol': eps_sol,
            'size_d': size_d,
            'm_e': m_e,
            'm_h': m_h,
            'T': T,
            'mu_e': mu_e,
        },
    )

    return file_id


file_id = '1SPdBq3kQwOllTluF-gw6Q'
#file_id = save_be_data()

data = load_data('extra/keldysh/be_ns_%s' % file_id, globals())

eps_vec, N_eps = array(eps_vec), len(eps_vec)

be_cou_vec[:] = data[0].reshape(N_eps, N_ns)
be_qcke_vec[:] = data[1].reshape(N_eps, N_ns)
be_ke_vec[:] = data[2].reshape(N_eps, N_ns)
be_hn_vec[:] = data[3].reshape(N_eps, N_ns)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_ns)
]

print('Bohr radius: %f nm' % sys_sol.exc_bohr_radius())

for nd, c in zip(range(N_ns), colors):
    ax[0].semilogx(
        eps_vec / eps_sol,
        be_ke_vec[:, nd] * 1e3,
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.0,
    )

    ax[0].semilogx(
        eps_vec / eps_sol,
        be_hn_vec[:, nd] * 1e3,
        color=c,
        linestyle='--',
        dashes=(0.2, 2.),
        dash_capstyle='round',
        linewidth=1.3,
    )

    ax[0].semilogx(
        eps_vec / eps_sol,
        be_qcke_vec[:, nd] * 1e3,
        color=c,
        linestyle='-',
        linewidth=1.8,
        label=r'$%d$s' % (nd + 1),
    )

    ax[1].semilogx(
        eps_vec / eps_sol,
        be_ke_vec[:, nd] * 1e3,
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.0,
    )

    ax[1].semilogx(
        eps_vec / eps_sol,
        be_hn_vec[:, nd] * 1e3,
        color=c,
        linestyle='--',
        dashes=(0.2, 2.),
        dash_capstyle='round',
        linewidth=1.3,
    )

    ax[1].semilogx(
        eps_vec / eps_sol,
        be_qcke_vec[:, nd] * 1e3,
        color=c,
        linestyle='-',
        linewidth=1.8,
    )

ax[0].legend(loc=0)

ax[0].set_ylabel(r'$\mathcal{E}$ (meV)')
ax[0].yaxis.set_label_coords(-0.25, 0.5)
ax[0].set_ylim(-1e3, 0)
ax[0].set_yticks([-1e3, -0.75e3, -0.5e3, -0.25e3, 0])
ax[0].set_yticklabels(['$%d$' % d for d in ax[0].get_yticks()])

ax[0].set_xlabel(r'$d^* / d$')
ax[0].set_xlim(eps_vec[0] / eps_sol, eps_vec[-1] / eps_sol)
ax[0].set_xticks([1])
ax[0].set_xticklabels(['$%d$' % d for d in ax[0].get_xticks()])
ax[0].set_xticks([2, 4, 8], minor=True)
ax[0].set_xticklabels(['$%d$' % d for d in ax[0].get_xticks(minor=True)],
                      minor=True)

#ax[1].set_ylabel(r'$\mathcal{E}$ / meV')
#ax[1].yaxis.set_label_coords(1.35, 0.5)
ax[1].yaxis.tick_right()
ax[1].set_ylim(-85, -70)
ax[1].set_yticks([-85, -80, -75, -70])
ax[1].set_yticklabels(['$%d$' % d for d in ax[1].get_yticks()])

ax[1].set_xlabel(r'$d^* / d$')
ax[1].set_xlim(eps_vec[0] / eps_sol, 4)
ax[1].set_xticks([1])
ax[1].set_xticklabels(['$%d$' % d for d in ax[1].get_xticks()])
ax[1].set_xticks([2, 4], minor=True)
ax[1].set_xticklabels(['$%d$' % d for d in ax[1].get_xticks(minor=True)],
                      minor=True)

rect = Rectangle(
    (ax[1].get_xlim()[0], ax[1].get_ylim()[0]),
    ax[1].get_xlim()[1] - ax[1].get_xlim()[0],
    ax[1].get_ylim()[1] - ax[1].get_ylim()[0],
    fill=False,
)

collection = PatchCollection(
    [rect],
    linestyle='-',
    linewidth=0.8,
    edgecolor='k',
    facecolor='#FFFFFF00',
    alpha=0.7,
)
ax[0].add_collection(collection)

plt.tight_layout()
#plt.setp(ax[0].get_xticklabels(), visible=False)
#plt.setp(ax[0].get_xticklabels(minor=True), visible=False)
fig.subplots_adjust(wspace=0.01)

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'be_ns_v1')

plt.show()

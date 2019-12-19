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


def ke_k_pot(u, x, sys):
    if abs(sys.eps_mat - sys.eps_r) < 1e-8:
        pot = 1
    else:
        pot = 1 / tanh(u * sys.size_d * 0.5 / x +
                       eta_func(sys.eps_mat, sys.eps_r))

    return -special.j0(u) * pot / x / sys.eps_mat * sys.c_aEM * sys.c_hbarc


N_x, N_eps = 128, 3

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.22, 0.41, 294  # K

eps_vec = eps_sol / array([1.0, 0.5, 0.2])

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_vec = [system_data(m_e, m_h, eps_sol, T, size_d, eps) for eps in eps_vec]


def integ_pot(pot_func, x, sys):
    return sum([
        quad(
            pot_func,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, sys),
        )[0] for n in arange(1e4)
    ])


x_vec = logspace(log10(2e-3), 2, N_x)

integ_ke_args = [(
    integ_pot,
    ke_k_pot,
    x,
    sys,
) for x, sys in itertools.product(x_vec, sys_vec)]

pool = multiprocessing.Pool(multiprocessing.cpu_count())
y_vec = array(time_func(
    pool.starmap,
    time_func,
    integ_ke_args,
)).reshape((N_x, N_eps))
pool.terminate()

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, eps_vec.size)
]

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    energy_norm = sys.c_aEM * sys.c_hbarc / sys.eps_mat / sys.size_d
    ax[0].semilogx(
        x_vec,
        -1 / (y_vec[:, n_eps] / energy_norm * x_vec),
        '-',
        color=c,
        label=r'$d^* / d$: $%d$' % (size_d_eff(sys) / size_d),
    )
    ax[0].axhline(y=eps_sol / sys.eps_mat, color=c, linewidth=0.7)

ax[0].set_yticks([0, 1])
ax[0].yaxis.set_label_coords(-0.02, 0.5)

ax[0].set_ylabel(r'$\epsilon(r) / \epsilon$')
ax[0].set_xlabel(r'$r / d$')

ax[0].set_xlim(x_vec[0], x_vec[-1])

ax[0].legend()

plt.tight_layout()

fig.subplots_adjust(wspace=0)

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'pot_comparison_eps_v1')

plt.show()

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


N_x, N_eps = 48, 4

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.22, 0.41, 294  # K

eps_vec = logspace(0, 1, N_eps)

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
        )[0] for n in arange(0.5e4)
    ])


#x_vec = linspace(5e-3, 4 * eps / eps_sol * size_d, 256)
x_vec = logspace(log10(2e-3), log10(80), N_x)

integ_ke_args = [(integ_pot, ke_k_pot, x, sys)
                 for x, sys in itertools.product(x_vec, sys_vec)]

pool = multiprocessing.Pool(multiprocessing.cpu_count())
y_vec = array(time_func(pool.starmap, time_func, integ_ke_args)).reshape(
    (N_x, N_eps))
pool.terminate()

x_short_vec = logspace(log10(x_vec[0]), 0, N_x)
x_long_vec = logspace(0, log10(x_vec[-1]), N_x)
cou_short_vec = -1 / x_short_vec / sys_sol.eps_r

cou_long_vec = array([-1 / x_long_vec * size_d_eff(sys)
                      for sys in sys_vec]).reshape(N_eps, N_x).T

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, eps_vec.size)
]

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    eps = sys.eps_mat

    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / eps / size_d
    ax[0].axvline(
        x=size_d_eff(sys) / size_d,
        linestyle='--',
        color=c,
        linewidth=0.7,
    )

    ax[0].semilogx(
        x_vec / size_d,
        y_vec[:, n_eps] / energy_norm,
        '-',
        color=c,
        #label=r'$d^*/d$: $%.1f$' % (size_d_eff(sys) / size_d),
    )
    ax[0].semilogx(
        x_long_vec / size_d,
        cou_long_vec[:, n_eps],
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.0,
        color=c,
    )
"""
ax[0].semilogx(
    x_short_vec / size_d_eff(sys_sol),
    cou_short_vec,
    linestyle='--',
    dashes=(3., 5.),
    dash_capstyle='round',
    linewidth=1.5,
    color='b',
)
"""

ax[0].axvline(x=1.0, color='m', linewidth=0.9)
ax[1].axvline(x=1.0, color='m', linewidth=0.9)
ax[0].axhline(y=0, color='k', linewidth=0.7)

real_eps_sol = -1 / (y_vec[0, 0] / energy_norm * x_vec[0])
for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    eps = sys.eps_mat
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / eps / size_d
    ax[1].semilogx(
        x_vec,
        -1 / (y_vec[:, n_eps] / energy_norm * x_vec),
        '-',
        color=c,
    )
    #ax[1].axhline(y=real_eps_sol / eps, color=c, linewidth=0.7)

ax[1].axhline(y=real_eps_sol, color='b', linewidth=0.7)

ax[0].set_yticks([0])
ax[0].set_yticklabels(['$0$'])
ax[0].yaxis.set_label_coords(-0.02, 0.5)

ax[1].set_yticks([real_eps_sol])
ax[1].set_yticklabels([r'$1$'])
ax[1].yaxis.set_label_coords(1.12, 0.5)

ax[0].set_ylabel(r'$V_{RK}\left(\frac{r}{d}\right)$')
ax[1].set_ylabel(r'$\epsilon\left(\frac{r}{d}\right)$')
ax[0].set_xlabel(r'$r / d$')
ax[1].set_xlabel(r'$r / d$')

#ax[0].legend()

ax[0].set_ylim(-10, 0)
ax[1].yaxis.tick_right()
ax[0].set_xlim(1e-1, x_vec[-1])
ax[1].set_xlim(x_vec[0], x_vec[-1])

plt.tight_layout()

fig.subplots_adjust(wspace=0)

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'pot_comparison_v3')

plt.show()

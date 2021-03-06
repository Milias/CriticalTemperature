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


def qc_x_pot(x, size_d):
    return -(3 * special.k0(2 * pi * x / size_d) +
             2 * pi * x / size_d * special.k1(2 * pi * x / size_d)) / size_d


def clke_k_pot(u, x, size_d, eta):
    u2 = u * size_d / x
    return 1.0 / tanh(0.5 * u + eta)


def qc_k_pot(u, x, size_d):
    u2 = u * size_d / x
    return u2 * (3 * u2**2 + 20 * pi**2) / (u2**2 + 4 * pi**2)**2


def qccougg_k_pot(u, x, size_d, *args):
    u2 = u * size_d / x
    return 32 * pi**4 * (u2 - 1.0 + exp(-u2)) / (u2 * (u2**2 + 4 * pi**2))**2


def qckegg_k_pot(u, x, size_d, eta):
    u2 = u * size_d / x

    if isinstance(u, float):
        if u > 700:
            return 32 * pi**4 * (u2 - 2 * sinh(eta)) / (u2 *
                                                        (u2**2 + 4 * pi**2))**2

    return 32 * pi**4 * (u2 - (sinh(u2 + eta) * sinh(eta) + sinh(u2 + eta) *
                               sinh(eta) - 2 * sinh(eta) * sinh(eta)) /
                         sinh(u2 + 2 * eta)) / (u2 * (u2**2 + 4 * pi**2))**2


size_d = 1.0  # nm
eps_sol = 1.0
#eps    = (1 + 1e-1) * eps_sol
eps = 10.0
eta = 0.5 * log((eps + eps_sol) / (eps - eps_sol))

m_e, m_h, T = 0.22, 0.41, 294  # K
sys = system_data(m_e, m_h, eps, T)

u_vec = linspace(1e-3, 12, 256)


def plot_k_pot_comp(x, c):
    factor = 2 * pi * sys.c_aEM * sys.c_hbarc

    qc_vec = factor * qc_k_pot(u_vec, x, size_d) / eps
    qckegg_vec = factor * qckegg_k_pot(u_vec, x, size_d, eta) / eps
    clke_vec = factor * clke_k_pot(u_vec, x, size_d, eta) / eps
    HN_vec = factor * 2 / (u_vec * size_d / x + 2 * eps_sol / eps) / eps
    HN2_vec = factor * 2 / (u_vec * 2 / 3 * size_d / x +
                            2 * eps_sol / eps) / eps

    full_ke_vec = qckegg_vec + qc_vec

    ax[0].plot(
        u_vec,
        full_ke_vec,
        color=c,
        linewidth=2.0,
        label=(r'$r = %.1f d$' % x) if x < 1 else
        ((r'$r = %.0f d$' % x) if x > 1 else (r'$r = d$')),
        #label=r'${\textrm{qc,RK}}$',
    )
    ax[0].plot(
        u_vec,
        qc_vec,
        color=c,
        linestyle='--',
        dashes=(0.2, 4.),
        dash_capstyle='round',
        linewidth=1.7,
        #label=r'${\textrm{qc}}$',
    )
    ax[0].plot(
        u_vec,
        qckegg_vec,
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.4,
        #label=r'${\textrm{qc,RK}}^{(\gg)}$',
    )

    ax[0].set_yticks([0, factor / eps_sol])
    ax[0].set_yticklabels([
        r'$0$',
        #r'$\frac{e^2}{2\epsilon_0\epsilon_{sol}}$',
        r'$\epsilon_{sol}^{-1}$',
    ])
    #ax[0].set_ylabel('$k ~ V(k)$ / eV')
    ax[0].set_ylabel('$k ~ V(k)$')
    ax[0].set_xlabel(r'$kd$')
    ax[0].yaxis.set_label_coords(-0.02, 0.5)

    ax[0].legend()

    ax[0].set_ylim(-0.3, 1.1 * factor / eps_sol)
    ax[0].set_xlim(u_vec[0], u_vec[-1])

    ax[0].axhline(
        y=0,
        color='k',
        linewidth=0.5,
    )


x_vec = logspace(-1, 1, 3)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, x_vec.size)
]

for x, c in zip(x_vec, colors):
    plot_k_pot_comp(x, c)

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'k_pot_comp')
plt.show()

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
    u2_low = u2[u2 < 700]
    u2_high = u2[u2 >= 700]

    result = zeros_like(u)

    result[u2 < 700] = 32 * pi**4 * (
        u2_low -
        (sinh(u2_low + eta) * sinh(eta) + sinh(u2_low + eta) * sinh(eta) -
         2 * sinh(eta) * sinh(eta)) / sinh(u2_low + 2 * eta)) / (
             u2_low * (u2_low**2 + 4 * pi**2))**2

    result[u2 >= 700] = 32 * pi**4 * (u2_high - 2 * sinh(eta)) / (
        u2_high * (u2_high**2 + 4 * pi**2))**2

    return result


size_d = 1.0  # nm
eps_sol = 1.0
#eps = (1 + 1e-1) * eps_sol
eps = 3.0
eta = 0.5 * log((eps + eps_sol) / (eps - eps_sol))

m_e, m_h, T = 0.22, 0.41, 294  # K
sys = system_data(m_e, m_h, eps, T)

#u_vec = linspace(1e-3, 11, 256)
u_vec = logspace(-3, 3, 1 << 10)


def plot_k_pot_comp():
    factor = 2 * pi * sys.c_aEM * sys.c_hbarc

    qc_vec = factor * qc_k_pot(u_vec, 1, size_d) / eps
    qckegg_vec = factor * qckegg_k_pot(u_vec, 1, size_d, eta) / eps
    clke_vec = factor * clke_k_pot(u_vec, 1, size_d, eta) / eps
    HN_vec = factor * 2 / (u_vec * size_d + 2 * eps_sol / eps) / eps

    full_ke_vec = qckegg_vec + qc_vec

    c = matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([1, 0.8, 0.8]))

    ax[0].axhline(y=factor / eps, color='k', linewidth=0.6)

    ax[0].axvline(
        x=1,
        linestyle='--',
        color='m',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.8,
    )

    ax[0].semilogx(
        u_vec,
        full_ke_vec,
        color=c,
        linestyle='-',
        linewidth=1.8,
        label=r'$V_{qc}^{RK}$',
    )

    ax[0].semilogx(
        u_vec,
        HN_vec,
        color=c,
        linestyle='--',
        dashes=(0.8, 4.),
        dash_capstyle='round',
        linewidth=1.0,
        label=r'$V^{\mathcal{H}N}$',
    )

    ax[0].semilogx(
        u_vec,
        clke_vec,
        color=c,
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.4,
        label=r'$V^{RK}$',
    )

    ax[0].set_yticks([0, factor / eps, factor / eps_sol])
    ax[0].set_yticklabels([
        r'$0$',
        r'$1$',
        r'$\frac{d^*}{d}$',
    ])
    ax[0].set_ylabel(r'$V(k) ~/~ V^{C}(k)$')
    ax[0].set_xlabel(r'$kd$')
    ax[0].yaxis.set_label_coords(-0.05, 0.5)

    ax[0].legend()

    ax[0].set_ylim(0, 1.1 * factor / eps_sol)
    ax[0].set_xlim(u_vec[0], u_vec[-1])


plot_k_pot_comp()

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'k_pot_comp_v2')
plt.show()

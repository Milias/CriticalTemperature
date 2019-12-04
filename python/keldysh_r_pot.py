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


def qccougg_k_pot(u, x, size_d, *args):
    u *= size_d / x
    return 32 * pi**4 * (u - 1.0 + exp(-u)) / (u * (u**2 + 4 * pi**2))**2


def clke_k_pot(u, x, size_d, eta):
    u *= size_d / x
    return 1.0 / tanh(0.5 * u + eta)


def qckegg_k_pot(u, x, size_d, eta):
    u *= size_d / x

    if u > 700:
        return 32 * pi**4 * (u - 2 * sinh(eta)) / (u * (u**2 + 4 * pi**2))**2

    return 32 * pi**4 * (u - (sinh(u + eta) * sinh(eta) + sinh(u + eta) *
                              sinh(eta) - 2 * sinh(eta) * sinh(eta)) /
                         (sinh(u + 2 * eta))) / (u * (u**2 + 4 * pi**2))**2


size_d = 1e-2  # nm
eps_sol = 1.0
eps = 30.0

eta = 0.5 * log((eps + eps_sol) / (eps - eps_sol))
m_e, m_h, T = 0.22, 0.41, 294  # K

sys = system_data(m_e, m_h, eps_sol, T, size_d, eps)


def transf_cou(u, *args):
    return special.j0(u) * qccougg_k_pot(u, *args)


def integ_cou_pot(x):
    return -sum([
        quad(
            transf_cou,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, size_d, eta),
        )[0] for n in arange(1e4)
    ]) / x / eps_sol


def plot_qccou_r_pot():
    x_vec = logspace(log10(1e-3), log10(3), 256)
    pool = multiprocessing.Pool(32)
    gg_vec = array(pool.map(integ_cou_pot, x_vec))
    qc_vec = qc_x_pot(x_vec, size_d) / eps_sol
    full_vec = gg_vec + qc_vec

    ax[0].axhline(y=0, color='k', linewidth=0.7)

    ax[0].plot(
        x_vec,
        qc_vec,
        'g',
        linestyle='--',
        dashes=(0.8, 4.),
        dash_capstyle='round',
        linewidth=2.5,
        label=r'$V_{\textrm{qc}}(r)$',
    )
    ax[0].plot(
        x_vec,
        gg_vec,
        'b',
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.5,
        label=r'$V_{\textrm{qc,Cou}}^{(\gg)}(r)$',
    )

    ax[0].plot(
        x_vec,
        full_vec,
        'r-',
        linewidth=2.0,
        label=r'$V_{\textrm{qc,Cou}}(r)$',
    )

    ax[0].set_xticks([0, 1, 2, 3])
    ax[0].set_yticks([0])
    ax[0].yaxis.set_label_coords(-0.01, 0.5)

    ax[0].set_ylabel('$V(r)$')
    ax[0].set_xlabel('$r ~ d^{-1}$')

    ax[0].legend()

    ax[0].set_ylim(-3, 0)
    ax[0].set_xlim(0, x_vec[-1])

    plt.tight_layout()

    plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
                'r_qccou_pot')


def transf_ke(u, *args):
    return special.j0(u) * qckegg_k_pot(u, *args)


def integ_ke_pot(x):
    return -sum([
        quad(
            transf_ke,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, size_d, eta),
        )[0] for n in arange(1e4)
    ]) / x / eps


def plot_qcke_r_pot():
    x_vec = logspace(log10(1e-3), log10(60), 256)
    pool = multiprocessing.Pool(32)
    gg_vec = array(pool.map(integ_ke_pot, x_vec))
    qc_vec = qc_x_pot(x_vec, size_d) / eps
    full_vec = gg_vec + qc_vec

    pool.terminate()

    approx_vec = array(
        exciton_pot_ke_vec(
            sys.size_d,
            sys.eps_mat,
            x_vec,
            sys,
        )) / (sys.c_aEM * sys.c_hbarc)
    """
    approx_vec = array(exciton_pot_kelr_vec(
        x_vec,
        sys,
    )) / (sys.c_aEM * sys.c_hbarc)
    """

    be_qcke = time_func(
        plasmon_det_zero_qcke,
        1 << 9,
        -1e2,
        sys.get_mu_h(-1e2),
        sys,
        -6e-3,
    )

    ax[0].axhline(y=0, color='k', linewidth=0.7)
    ax[0].axhline(y=be_qcke, color='m', linewidth=0.9)

    ax[0].plot(
        x_vec,
        qc_vec,
        'g',
        linestyle='--',
        dashes=(0.8, 4.),
        dash_capstyle='round',
        linewidth=2.5,
        label=r'$V_{\textrm{qc}}(r)$',
    )
    ax[0].plot(
        x_vec,
        gg_vec,
        'b',
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.5,
        label=r'$V_{\textrm{qc,RK}}^{(\gg)}(r)$',
    )

    ax[0].plot(
        x_vec,
        full_vec,
        'r-',
        linewidth=2.0,
        label=r'$V_{\textrm{qc,RK}}(r)$',
    )

    ax[0].plot(
        x_vec,
        -1 / sys.eps_r / x_vec,
        'g--',
        linewidth=1.0,
        label=r'$V_{\sim}(r)$',
    )

    ax[0].plot(
        x_vec,
        approx_vec,
        'r--',
        linewidth=1.0,
        label=r'$V_{\sim}(r)$',
    )

    ax[0].set_xticks([0, 1, 2, 3])
    ax[0].set_yticks([0])
    ax[0].yaxis.set_label_coords(-0.01, 0.5)

    ax[0].set_ylabel('$V(r)$')
    ax[0].set_xlabel('$r ~ d^{-1}$')

    ax[0].legend()

    ax[0].set_ylim(-3, 0)
    ax[0].set_xlim(0, x_vec[-1])

    plt.tight_layout()

    plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
                'r_qcke_pot_2')


#plot_qccou_r_pot()
#plt.cla()
plot_qcke_r_pot()
plt.show()

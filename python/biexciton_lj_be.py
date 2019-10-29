from common import *


def root_be(eps_1, eps_2, N_k, mu_e, mu_h, be_exc, func):
    if eps_2 is None:
        sys = system_data(m_e, m_h, eps_1, T)
    else:
        sys = system_data(m_e, m_h, eps_2, T)
        sys.size_d = 1.37
        sys.eps_mat = eps_1

    return func(N_k, mu_e, mu_h, sys, -5e-2) - be_exc


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

be_exc = -193e-3  # eV
be_biexc_exp = -45e-3  # eV
surf_area = 326.4  # nm^2

N_k = 1 << 9
N_n = 64

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

mu_e = -1e2
mu_h = sys.get_mu_h(mu_e)

eps_r = root_scalar(
    root_be,
    args=(None, N_k, mu_e, mu_h, be_exc, plasmon_det_zero_ht),
    method='brentq',
    bracket=(1.0, 10.0),
).root

sys = system_data(m_e, m_h, eps_r, T)

be_exc_ht = time_func(plasmon_det_zero_ht, N_k, mu_e, mu_h, sys, -5e-2)
print('[Coulomb] eps_r: %f, eb: %f eV' % (eps_r, be_exc_ht))

param_c12 = time_func(biexciton_c12_lj, be_exc, be_biexc_exp, sys)
print('[LJ] C_12: %f eV nm^-12' % param_c12)

z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)
print('[LWL] E_B^(Sat): %f eV' % z_sys_lwl)

n_max = sys.density_ideal(-1 / sys.beta)
n_vec = logspace(-3, log10(n_max * surf_area), N_n) / surf_area
n_lwl_vec = logspace(-2, 0, N_n) / surf_area

mu_e_vec = mu_e_vec = array([sys.mu_ideal(n) for n in n_vec])
mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])

ls_vec = array([sys.ls_ideal(n) for n in n_lwl_vec])

be_exc_lwl_vec = array(
    [plasmon_det_zero_lwl(
        N_k,
        ls,
        sys,
        z_sys_lwl,
    ) for ls in ls_vec])

be_biexc_scr = array(
    [biexciton_be_lj(
        param_c12,
        be,
        sys,
    ) for be in be_exc_lwl_vec])

ax[0].semilogx(n_lwl_vec * surf_area, be_biexc_scr, 'r-')
ax[0].semilogx(n_lwl_vec * surf_area, be_exc_lwl_vec, 'r:')

ax[0].legend(loc=0)

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Biexcitons/%s.pdf' %
            'biexciton_lj_be_n_q')

plt.show()

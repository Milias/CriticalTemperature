from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]) * 2)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

N_k = 1 << 10
be_exc = -193e-3  # eV
m_e, m_h, eps_r, T = 0.22, 0.41, 1.442, 294  # K
sys = system_data(m_e, m_h, eps_r, T)
sys.eps_mat = 8.0
sys.size_d = 1.37  # nm

mu_e = 1e-1
mu_h = sys.get_mu_h(mu_e)
"""
be_exc_ke = plasmon_det_zero_ke(N_k, mu_e, mu_h, sys, 2 * be_exc)
print(be_exc_ke)
"""

k_vec = linspace(1e-4, 3e1, 1 << 10)
wk_vec = array([[0.0, k] for k in k_vec]).flatten()
ke_v = array(plasmon_green_ke_v(k_vec, mu_e, mu_h, sys))
green_v = array(plasmon_green_v(wk_vec, mu_e, mu_h, sys))
ht_v = array(plasmon_green_ht_v(wk_vec, mu_e, mu_h, sys))

ax[0].semilogy(k_vec, -ke_v, 'r-', label='Keldysh')
ax[0].semilogy(k_vec, -real(green_v), 'b-', label=r'$T=0$')
ax[0].semilogy(k_vec, -real(ht_v), 'g-', label='Classical')

#"""
plt.tight_layout()
plt.legend()

plt.savefig('/storage/Reference/Work/University/PhD/Excitons/%s.pdf' %
            'exciton_be_ke_green_comp')

plt.show()
#"""

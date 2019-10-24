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

be_exc = -193e-3  # eV
surf_area = 326.4  # nm^2
"""
    http://www.ddbst.com/en/EED/PCP/DEC_C89.php

    Hexane dielectric constant:

    T           P               eps_r
    298.15 	101426.000 	1.99550

    Source:
        Mopsik F.I.: Dielectric Constant of n-Hexane as a Function of Temperature, Pressure and Density. J.Res. NBS Sect.A 71 (1967) 287-292
"""

with open('extra/biexcitons/permittivity.json', 'r') as f:
    eps_map = json.load(f)

N_k = eps_map['N_k']
size_d = eps_map['size_d']

eps_sol = eps_map['eps_sol']
eps_mat = eps_map['eps_mat']
eps_r = eps_map['eps_r']

m_e, m_h, T = 0.22, 0.41, 294  # K

sys_cou = system_data(m_e, m_h, eps_r, T)

mu_e_cou = -1e2
mu_h_cou = sys_cou.get_mu_h(mu_e_cou)

print('[** Sanity check **]')
be_exc_ht = plasmon_det_zero_ht(N_k, mu_e_cou, mu_h_cou, sys_cou, -5e-2)
print('[Coulomb] eps: %f, E_B: %f eV' % (eps_r, be_exc_ht))

sys_ke = system_data(m_e, m_h, eps_sol, T)
sys_ke.eps_mat = eps_mat
sys_ke.size_d = size_d

be_exc_ke = plasmon_det_zero_ke(N_k, mu_e_cou, mu_h_cou, sys_ke, -5e-2)
print('[Keldysh] eps_mat: %f, eps_sol: %f, E_B: %f eV' %
      (eps_mat, eps_sol, be_exc_ke))

n_max = sys_cou.density_ideal(-1 / sys_cou.beta)
n_vec = logspace(-3, log10(n_max * surf_area), 48) / surf_area
n_lwl_vec = logspace(-3, 2, 48) / surf_area

mu_e_vec = mu_e_vec = array([sys_cou.mu_ideal(n) for n in n_vec])
mu_h_vec = array([sys_cou.get_mu_h(mu_e) for mu_e in mu_e_vec])

ls_vec = array([sys_cou.ls_ideal(n) for n in n_lwl_vec])

z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys_cou.sys_ls, sys_cou)
print('[LW-L] E_B^(Sat): %f eV' % z_sys_lwl)

be_exc_lwl_vec = array([
    time_func(plasmon_det_zero_lwl, N_k, ls, sys_cou, z_sys_lwl)
    for ls in ls_vec
])

be_exc_cou_vec = array([
    time_func(plasmon_det_zero_ht, N_k, mu_e, mu_h, sys_cou, z_sys_lwl)
    for (mu_e, mu_h) in zip(mu_e_vec, mu_h_vec)
])

be_exc_ke_vec = array([
    time_func(plasmon_det_zero_ke, N_k, mu_e, mu_h, sys_ke, z_sys_lwl)
    for (mu_e, mu_h) in zip(mu_e_vec, mu_h_vec)
])

ax[0].set_xlabel(r'$\langle N_q \rangle$')
ax[0].set_ylabel(r'$E_B(n_q)$ (meV)')

ax[0].semilogx(n_lwl_vec * surf_area, 1e3 * be_exc_lwl_vec, 'r:')

ax[0].semilogx(n_vec * surf_area, 1e3 * be_exc_cou_vec, 'r--', label='Coulomb')
ax[0].semilogx([n_vec[-1] * surf_area], [1e3 * be_exc_cou_vec[-1]], 'ro')

ax[0].semilogx(n_vec * surf_area, 1e3 * be_exc_ke_vec, 'r-', label='Keldysh')
ax[0].semilogx([n_vec[-1] * surf_area], [1e3 * be_exc_ke_vec[-1]], 'ro')

ax[0].axhline(
    y=be_exc * 1e3,
    color='k',
    linestyle='--',
    label=r'$\langle N_q\rangle$: %d' % 0,
    linewidth=0.9,
)

ax[0].set_xlim(n_lwl_vec[0] * surf_area, n_lwl_vec[-1] * surf_area)

plt.tight_layout()
plt.legend()

plt.savefig('/storage/Reference/Work/University/PhD/Excitons/%s.pdf' %
            'exciton_be_ke_density')

plt.show()

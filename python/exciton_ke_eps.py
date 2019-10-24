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

N_k = 1 << 11
be_exc = -193e-3  # eV
size_d = 1.37  # nm
"""
    http://www.ddbst.com/en/EED/PCP/DEC_C89.php

    Hexane dielectric constant:

    T           P               eps_r
    298.15 	101426.000 	1.99550

    Source:
        Mopsik F.I.: Dielectric Constant of n-Hexane as a Function of Temperature, Pressure and Density. J.Res. NBS Sect.A 71 (1967) 287-292
"""

m_e, m_h, eps_r, T = 0.22, 0.41, 6.36, 294  # K


def root_be(eps_1, eps_2, N_k, mu_e, mu_h, be_exc, func):
    if eps_2 is None:
        sys = system_data(m_e, m_h, eps_1, T)
    else:
        sys = system_data(m_e, m_h, eps_2, T)
        sys.size_d = 1.37
        sys.eps_mat = eps_1

    return func(N_k, mu_e, mu_h, sys, -5e-2) - be_exc


sys = system_data(m_e, m_h, eps_r, T)

mu_e = -1e2
mu_h = sys.get_mu_h(mu_e)

print('mu_e: %f, mu_h: %f\n' % (mu_e, mu_h))
eps = root_scalar(
    root_be,
    args=(None, N_k, mu_e, mu_h, be_exc, plasmon_det_zero_ht),
    method='brentq',
    bracket=(1.0, 10.0),
).root

sys = system_data(m_e, m_h, eps, T)

be_exc_ht = plasmon_det_zero_ht(N_k, mu_e, mu_h, sys, -5e-2)
print('[Coulomb] eps: %f, eb: %f eV' % (eps, be_exc_ht))

eps_sol = 1.9955
eps_mat = 6.0

eps_mat = root_scalar(
    root_be,
    args=(eps_sol, N_k, mu_e, mu_h, be_exc, plasmon_det_zero_ke),
    method='brentq',
    bracket=(eps_sol * 1.05, 10.0),
).root

sys = system_data(m_e, m_h, eps_sol, T)
sys.eps_mat = eps_mat
sys.size_d = size_d

be_exc_ke = plasmon_det_zero_ke(N_k, mu_e, mu_h, sys, -1e-1)
print('[Keldysh] eps_mat: %f, eps_sol: %f, be: %f eV' %
      (eps_mat, eps_sol, be_exc_ke))

with open('extra/biexcitons/permittivity.json', 'w+') as f:
    json.dump(
        {
            'eps_mat': eps_mat,
            'eps_sol': eps_sol,
            'eps_r': eps,
            'N_k': N_k,
            'size_d': size_d
        }, f)
"""
plt.tight_layout()
plt.legend()

plt.savefig('/storage/Reference/Work/University/PhD/Excitons/%s.pdf' %
            'exciton_be_ke_eps')

plt.show()
"""

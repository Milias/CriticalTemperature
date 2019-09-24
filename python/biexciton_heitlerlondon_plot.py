from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

file_id = '939Tcd2ySviOVirRWCJZ5w'
file_id = 'Zk8eW8jhTueqjfxVFq865Q'  # 1024 [1e-3, 128] log

data = load_data('extra/biexcitons/eff_pot_vec_%s' % file_id, globals())
param_vec = data[:, 0]
pot_vec = data[:, 1:]

pol_a2 = 21 / 2**8
th_pol_cou = 2 * pol_a2 * eps_r * (2 * sys.c_hbarc**2 / eb_cou / sys.m_p)**1.5
param_c6 = 24 / (eb_cou**2 * sys.m_p**3) * (pol_a2 * eps_r * sys.c_hbarc**3)**2

"""
r_crossover = 8
pot_vec[:, 6] = pot_vec[:, 6] * (1 - exp(-r_crossover / param_vec)) - exp(
    -r_crossover / param_vec) * param_c6 / param_vec**6
"""

plt.plot(param_vec, pot_vec[:, 0], 'r:', label=r'$\Delta$')
plt.plot(param_vec, pot_vec[:, 1], 'b:', label='$J$')
plt.plot(param_vec, pot_vec[:, 2], 'g:', label='$J\'$')
plt.plot(param_vec, pot_vec[:, 3], 'k:', label='$K$')
plt.plot(param_vec, pot_vec[:, 4], 'm:', label='$K\'$')
plt.axhline(y=0, color='k', linewidth=0.3)

plt.plot(param_vec,
         sys.c_aEM * sys.c_hbarc / param_vec / sys.eps_r,
         'k--',
         label='Cou')
plt.plot(param_vec, -param_c6 / param_vec**6, 'm--', label=r'$C_6 / r^6$')
plt.plot(param_vec, pot_vec[:, 5], 'r-', label='$V_+(r_{BA})$')
plt.plot(param_vec, pot_vec[:, 6], 'b-', label='$V_-(r_{BA})$')

plt.legend(loc=0)
plt.tight_layout()
plt.ylim(-300e-3, 1.0)
plt.show()

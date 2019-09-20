from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

N_param = 1 << 7
#param_vec = logspace(-1, 1, N_param)
param_vec = linspace(1e-1, 16, N_param)
t0 = time.time()
pot_vec = array(
    [result_s(r).value for r in biexciton_eff_pot_vec(param_vec, sys)])
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_param, dt))

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
plt.plot(param_vec, pot_vec[:, 5], 'r-', label='$V_+(r_{BA})$')
plt.plot(param_vec, pot_vec[:, 6], 'b-', label='$V_-(r_{BA})$')

plt.legend(loc=0)
plt.tight_layout()
#plt.ylim(None, -amin(value_p_vec))
plt.show()

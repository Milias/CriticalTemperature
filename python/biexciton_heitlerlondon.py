from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

N_param = 1 << 7
#param_vec = logspace(-1, 1, N_param)
param_vec = linspace(1e-1, 16, N_param)
t0 = time.time()
result_Delta_vec = array(
    [result_s(r).total_value() for r in biexciton_Delta_r_vec(param_vec, sys)])
result_J_vec = array(
    [result_s(r).total_value() for r in biexciton_J_r_vec(param_vec, sys)])
result_Jp_vec = array(
    [result_s(r).total_value() for r in biexciton_Jp_r_vec(param_vec, sys)])
result_K_vec = array(
    [result_s(r).total_value() for r in biexciton_K_r_vec(param_vec, sys)])
result_Kp_vec = array(
    [result_s(r).total_value() for r in biexciton_Kp_r_vec(param_vec, sys)])
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_param, dt))

plt.plot(param_vec, result_Delta_vec, 'r:', label=r'$\Delta$')
plt.plot(param_vec, result_J_vec, 'b:', label='$J$')
plt.plot(param_vec, result_Jp_vec, 'g:', label='$J\'$')
plt.plot(param_vec, result_K_vec, 'k:', label='$K$')
plt.plot(param_vec, result_Kp_vec, 'm:', label='$K\'$')
plt.axhline(y=0, color='k', linewidth=0.3)

value_p_vec = sys.c_aEM * sys.c_hbarc / param_vec / sys.eps_r + (
    2 * result_J_vec + result_Jp_vec + 2 * result_Delta_vec * result_K_vec +
    result_Kp_vec) / (1 + result_Delta_vec**2)

value_m_vec = sys.c_aEM * sys.c_hbarc / param_vec / sys.eps_r + (
    2 * result_J_vec + result_Jp_vec - 2 * result_Delta_vec * result_K_vec -
    result_Kp_vec) / (1 - result_Delta_vec**2)

plt.plot(param_vec, value_p_vec, 'r-', label='$V_+(r_{BA})$')
plt.plot(param_vec, value_m_vec, 'b-', label='$V_-(r_{BA})$')

plt.legend(loc=0)
plt.tight_layout()
#plt.ylim(None, -amin(value_p_vec))
plt.show()

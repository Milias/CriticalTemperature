from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

N_param = 1 << 7
param_vec = logspace(-1, 1, N_param)
t0 = time.time()
result_Delta_vec = [result_s(biexciton_Delta_r(p, sys)) for p in param_vec]
result_J_vec = [result_s(biexciton_J_r(p, sys)) for p in param_vec]
result_Jp_vec = [result_s(biexciton_Jp_r(p, sys)) for p in param_vec]
result_K_vec = [result_s(biexciton_K_r(p, sys)) for p in param_vec]
result_Kp_vec = [result_s(biexciton_Delta_r(p, sys)) for p in param_vec]
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_param, dt))

value_p_vec = array([
    -2 * eb_cou + sys.c_aEM * sys.c_hbarc / r_BA / sys.eps_r +
    (2 * r_J.total_value() + r_Jp.total_value() +
     2 * r_D.total_value() * r_K.total_value() + r_Kp.total_value() * 0) /
    (1 + r_D.total_value()**2) for r_D, r_J, r_Jp, r_K, r_Kp, r_BA in zip(
        result_Delta_vec, result_J_vec, result_Jp_vec, result_K_vec,
        result_Kp_vec, param_vec)
])
value_m_vec = array([
    -2 * eb_cou + sys.c_aEM * sys.c_hbarc / r_BA / sys.eps_r +
    (2 * r_J.total_value() + r_Jp.total_value() -
     2 * r_D.total_value() * r_K.total_value() - r_Kp.total_value() * 0) /
    (1 - r_D.total_value()**2) for r_D, r_J, r_Jp, r_K, r_Kp, r_BA in zip(
        result_Delta_vec, result_J_vec, result_Jp_vec, result_K_vec,
        result_Kp_vec, param_vec)
])
"""
error_vec = array([
    sum([r.total_abs_error() for r in r_vec])
    for r_vec in zip(result_J_vec, result_Jp_vec)
])
"""

#print(value_vec)
#print(error_vec)
#print('%e' % amax(error_vec))

plt.plot(param_vec, value_p_vec, 'r-')
plt.plot(param_vec, value_m_vec, 'b-')
"""
plt.fill_between(
    param_vec,
    value_vec - error_vec,
    value_vec + error_vec,
    facecolor='green',
    interpolate=True,
)
"""
plt.tight_layout()
plt.ylim(None, -amin(value_p_vec))
plt.show()

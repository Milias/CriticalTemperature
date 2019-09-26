from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

N_param = 1 << 8
param_vec = logspace(-2, log10(32), N_param)
#param_vec = linspace(1e-3, 32, N_param)

"""
x1 = 1.0
x2 = 0.5
#x_vec = logspace(-2, 1, 10)
x_vec = linspace(1e-2, 10, 10)
th_vec = linspace(0, 2 * pi, 100)
plt.plot(
    x_vec,
    array([
        biexciton_Kp_r(x2, sys).total_value() for x2 in x_vec
        if print(x2) is None
    ]), 'r-')
plt.show()
exit()
"""

t0 = time.time()
pot_vec = array(
    [result_s(r).value for r in biexciton_eff_pot_vec(param_vec, sys)])
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_param, dt))

data = zeros((N_param, 8))
data[:, 0] = param_vec
data[:, 1:] = pot_vec

save_data(
    'extra/biexcitons/eff_pot_vec_%s' %
    base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2],
    data,
    {
        'm_e': m_e,
        'm_h': m_h,
        'T_vec': T_vec.tolist(),
        'eps_r': eps_r
    },
)

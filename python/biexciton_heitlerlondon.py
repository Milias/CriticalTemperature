from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

N_param = 1 << 10
#param_vec = logspace(-3, log10(128), N_param)
param_vec = linspace(1e-3, 128, N_param)
t0 = time.time()
pot_vec = array(
    [result_s(r).value for r in biexciton_eff_pot_vec(param_vec, sys)])
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_param, dt))

data = zeros((N_param, 8))
data[:, 0] = param_vec
data[:, 1:] = pot_vec[:]

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

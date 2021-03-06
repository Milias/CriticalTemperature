from common import *
import pyperclip

N_k = 1 << 10

surf_area = 326.4  # nm^2
eb_cou = 0.193
#m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
m_e, m_h, eps_r, T = 0.22, 0.41, 6.56, 294  # K
sys = system_data(m_e, m_h, eps_r, T)
eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
"""
Compute eps_r given the actual binding energy given by
the long wavelength method with ls -> 0.
"""
z_cou_sol = root(
    lambda eps_r: plasmon_det_zero_lwl(
        N_k,
        1e-8,
        system_data(m_e, m_h, eps_r[0], T),
        -1e-1,
    ) + eb_cou,
    [eps_r],
    method='hybr',
)

print(eps_r)

eps_r = z_cou_sol.x[0]
sys = system_data(m_e, m_h, eps_r, T)
z_cou_lwl = plasmon_det_zero_lwl(N_k, 1e-8, sys, -1e-1)

print(eps_r)
print(z_cou_lwl)

z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)

print('z_sys_lwl: %.3e' % z_sys_lwl)

T_vec = linspace(130, 350, 5)

values_list = []

for i, T in enumerate(T_vec):
    sys = system_data(m_e, m_h, eps_r, T)

    n_vec = logspace(-2, 2, 32) / surf_area
    mu_e_vec = array([sys.mu_ideal(n) for n in n_vec])
    ls_vec = array([sys.ls_ideal(n) for n in n_vec])

    eb_vec = array(time_func(plasmon_det_zero_lwl_v, N_k, ls_vec, sys))

    data_zeroes = zeros((mu_e_vec.size, ))

    data_zeroes[:] = mu_e_vec[:]
    values_list.append(array(data_zeroes.tolist()))

    data_zeroes[:] = eb_vec[:]
    values_list.append(array(data_zeroes.tolist()))

uuid_b64 = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]
pyperclip.copy(uuid_b64)

save_data(
    'extra/eb_lwl_temp_%s' %
    uuid_b64,
    values_list,
    {
        'm_e': m_e,
        'm_h': m_h,
        'T_vec': T_vec.tolist(),
        'eps_r': eps_r,
        'z_cou_lwl': z_cou_lwl,
    },
)

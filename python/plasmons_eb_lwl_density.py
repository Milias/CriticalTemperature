from common import *

N_k = 1 << 9

surf_area = 326.4  # nm^2
eb_cou = 0.193
#m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
m_e, m_h, eps_r, T = 0.22, 0.41, 6.0, 294  # K
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

T_vec = linspace(130, 350, 5)

values_list = []

for i, T in enumerate(T_vec):
    sys = system_data(m_e, m_h, eps_r, T)
    """
    mu_e_vec = linspace(-6.0, 18.0, 16) / sys.beta
    n_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    """

    n_vec = logspace(-4, log10(0.4), 32)
    mu_e_vec = array([sys.mu_ideal(n) for n in n_vec])

    ls_vec = array([0.5 * sys.ls_ideal(n) for n in n_vec])

    eb_vec = array(time_func(plasmon_det_zero_lwl_v, N_k, ls_vec, sys, -1e-2))

    data_zeroes = zeros((mu_e_vec.size, ))

    data_zeroes[:] = mu_e_vec[:]
    values_list.append(array(data_zeroes.tolist()))

    data_zeroes[:] = eb_vec[:]
    values_list.append(array(data_zeroes.tolist()))

save_data(
    'extra/eb_lwl_temp_%s' %
    base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2],
    values_list,
    {
        'm_e': m_e,
        'm_h': m_h,
        'T_vec': T_vec.tolist(),
        'eps_r': eps_r,
        'z_cou_lwl': z_cou_lwl,
    },
)

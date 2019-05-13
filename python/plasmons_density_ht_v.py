from common import *

N_k = 1 << 10

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

print(z_cou_sol.x)
print(eps_r)

eps_r = z_cou_sol.x[0]
sys = system_data(m_e, m_h, eps_r, T)

print(plasmon_det_zero_lwl(N_k, 1e-10, sys, -1e-1))

T_vec = linspace(294, 310, 1)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

n_exp_vec, cond_real, cond_imag, N_a_exp_vec, cond_err_real, cond_err_imag = load_data(
    'bin/cdse_platelet_data')

#n_vec = logspace(-3.5, 0.3, 1 << 7)
n_vec = n_exp_vec / surf_area

values_list = []

for c, (i, T) in zip(colors, enumerate(T_vec)):
    sys = system_data(m_e, m_h, eps_r, T)

    exc_list = time_func(plasmon_density_ht_v, n_vec, N_k, sys)

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    eb_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys,
                             eb_lim))

    data_zeroes = zeros((len(exc_list),))
    data_zeroes[2:] = n_vec

    values_list.append(array(data_zeroes.tolist()))
    values_list.append(exc_list[:])

    data_zeroes[2:] = eb_vec
    values_list.append(array(data_zeroes.tolist()))

save_data(
    'extra/mu_e_data_%s' %
    base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2],
    #(array([0, 0, *tuple(n_vec)]), exc_list, array([0, 0, *tuple(eb_vec)])),
    values_list,
    {
        'm_e': m_e,
        'm_h': m_h,
        'T_vec': T_vec.tolist(),
        'eps_r': eps_r
    },
)

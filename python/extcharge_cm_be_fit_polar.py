from common import *


def model_data_1g_d(loaded_data, size_vec, hwhm_vec, sys):
    def model_data_func(xdata, shift1, shift2, shift3, shift4):
        shift_vec = array([shift1, shift2, shift3, shift4])
        print('s: %.2f, %.2f, %.2f, %.2f meV' % tuple(shift_vec * 1e3))

        sum_model_all = []

        for ii, shift in enumerate(shift_vec):
            sys.size_Lx, sys.size_Ly = size_vec[ii]
            sys.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            data = array([
                exciton_PL_d_vec(xdata_ii - shift, nx, ny, sys)
                for nx, ny in states_vec
            ])

            sum_model = sum(data, axis=0)
            sum_model /= amax(sum_model)

            sum_model_all.extend(sum_model)

        return array(sum_model_all)

    return model_data_func


def model_data_1g(loaded_data, size_vec, hwhm_vec, sys):
    def model_data_func(xdata, gamma, shift1, shift2, shift3, shift4):
        shift_vec = array([shift1, shift2, shift3, shift4])

        print('Γ: %.2f meV' % (gamma * 1e3))
        print('s: %.2f, %.2f, %.2f, %.2f meV' % tuple(shift_vec * 1e3))

        sum_model_all = []

        for ii, shift in enumerate(shift_vec):
            sys.size_Lx, sys.size_Ly = size_vec[ii]
            sys.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            data = array([
                exciton_PL_vec(xdata_ii - shift, gamma, nx, ny, sys)
                for nx, ny in states_vec
            ])

            sum_model = sum(data, axis=0)
            sum_model /= amax(sum_model)

            sum_model_all.extend(sum_model)

        return array(sum_model_all)

    return model_data_func


def load_raw_PL_data(path, eV_max):
    raw = loadtxt(path)
    arg_max = raw[:, 1].argmax()
    xdata_eV = 1240.0 / raw[::-1, 0]
    xdata_eV -= xdata_eV[arg_max]

    xdata_eV_arg = abs(xdata_eV) < eV_max

    return array([
        xdata_eV[xdata_eV_arg],
        raw[xdata_eV_arg, 1] / amax(raw[xdata_eV_arg, 1]),
    ]).T


nmax = 1

states_vec = list(
    itertools.product(
        range(1, nmax + 1, 2),
        range(1, nmax + 1, 2),
    ))

N_states = len(states_vec)

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]
sizes_vec = [
    (29.32, 5.43),
    (26.11, 6.42),
    (25.4, 8.05),
    (13.74, 13.37),
]
hwhm_vec = [
    (3.3, 0.81),
    (3.34, 1.14),
    (2.9, 0.95),
    (2.17, 1.85),
]

sys = system_data(m_e, m_h, eps_sol, T, size_d, 1, 1, eps_sol)

loaded_data = array([
    load_raw_PL_data('extra/data/extcharge/PL-%s.txt' % label, 0.15)
    for label in labels_vec
])

concat_data = concatenate(loaded_data[:])

popt_list, pcov_list = [], []
gamma_p0 = [20e-3]
#gamma_p0 = []
shift_p0 = [-0.02] * len(sizes_vec)

popt, pcov = time_func(
    curve_fit,
    model_data_1g(loaded_data, sizes_vec, hwhm_vec, sys) if len(gamma_p0) > 0
    else model_data_1g_d(loaded_data, sizes_vec, hwhm_vec, sys),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=tuple(gamma_p0 + shift_p0),
    bounds=(tuple([0 for g in gamma_p0] + [-inf for s in shift_p0]), inf),
    method='trf',
)

print(popt)
print(pcov)

file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

save_data(
    'extra/extcharge/cm_be_polar_fit_params_%s' % file_id,
    [popt],
    extra_data={
        'labels_vec': labels_vec,
        'sizes_vec': sizes_vec,
        'hwhm_vec': hwhm_vec,
        'states_vec': states_vec,
        'pcov': pcov.tolist(),
        'n_gamma': len(gamma_p0),
    },
)

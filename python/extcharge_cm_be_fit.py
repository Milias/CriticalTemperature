from common import *


def distr_be(E, mu, sys):
    r = (E - mu) * sys.beta
    if abs(r) > 700:
        return 0

    return 1 / (exp(r) - 1)


def distr_mb(E, mu, sys):
    r = -(E - mu) * sys.beta
    if abs(r) > 700:
        return 0

    return exp(r)


def osc_strength(Lx, Ly, nx, ny, sys):
    return Lx * Ly / (nx * ny)**2


def state_energy(Lx, Ly, nx, ny, sys):
    return ((nx / Lx)**2 +
            (ny / Ly)**2) * (pi * sys.c_hbarc)**2 * 0.5 / (sys.m_e + sys.m_h)


def distr_sizes(x, L, hwhm):
    sigma = hwhm / sqrt(2 * log(2))
    return exp(-0.5 * ((x - L) / sigma)**2) / (sigma * sqrt(2 * pi))


def lorentz_distr(E, E0, gamma):
    return 0.5 * gamma / ((E - E0)**2 + 0.25 * gamma**2) / pi


def os_integrand(Lx, Ly, E, gamma, hwhm_x, hwhm_y, nx, ny, sys):
    return osc_strength(Lx, Ly, nx, ny, sys) * distr_sizes(
        Lx, sys.size_Lx, hwhm_x) * distr_sizes(
            Ly, sys.size_Ly, hwhm_y) * lorentz_distr(
                E, state_energy(Lx, Ly, nx, ny, sys), gamma)


def os_integrand_be(Lx, Ly, E, gamma, hwhm_x, hwhm_y, nx, ny, sys):
    E_S = state_energy(Lx, Ly, nx, ny, sys)

    return osc_strength(Lx, Ly, nx, ny, sys) * distr_sizes(
        Lx, sys.size_Lx, hwhm_x) * distr_sizes(
            Ly, sys.size_Ly, hwhm_y) * lorentz_distr(E, E_S, gamma) * distr_mb(
                E_S, 0, sys)


def os_integ(*args):
    return dblquad(
        os_integrand_be,
        0,
        float('inf'),
        0,
        float('inf'),
        args=args,
    )


def r_squared(data, model):
    data_avg = average(data)
    return 1 - sum((model - data)**2) / sum((data - data_avg)**2)


def model_data(size_Lx, size_Ly, hwhm_x, hwhm_y, sys, pool):
    def model_data_func(xdata, gamma, shift):
        print('Γ: %.2f meV, Δ: %.2f meV' % (gamma * 1e3, shift * 1e3))
        sys.size_Lx, sys.size_Ly = size_Lx, size_Ly

        starmap_args = [(
            E,
            gamma,
            hwhm_x,
            hwhm_y,
            nx,
            ny,
            sys,
        ) for (nx, ny), E in itertools.product(states_vec, xdata - shift)]

        model_data = array(time_func(
            pool.starmap,
            os_integ,
            starmap_args,
        )).reshape((N_states, xdata.size, 2))

        sum_model = sum(model_data[:, :, 0], axis=0)
        sum_model /= amax(sum_model)

        return sum_model

    return model_data_func


def model_data_all(loaded_data, size_vec, hwhm_vec, sys, pool):
    def model_data_func(xdata, gamma, shift1, shift2, shift3, shift4):
        print('Γ: %.2f meV' % (gamma * 1e3))
        shift_vec = array([shift1, shift2, shift3, shift4])

        sum_model_all = []

        for ii, shift in enumerate(shift_vec):
            sys.size_Lx, sys.size_Ly = size_vec[ii]

            xdata_ii = loaded_data[ii][:, 0]

            starmap_args = [(
                E,
                gamma,
                *hwhm_vec[ii],
                nx,
                ny,
                sys,
            ) for (nx, ny), E in itertools.product(
                states_vec,
                xdata_ii - shift,
            )]

            model_data = array(
                time_func(
                    pool.starmap,
                    os_integ,
                    starmap_args,
                )).reshape((N_states, xdata_ii.size, 2))

            sum_model = sum(model_data[:, :, 0], axis=0)
            sum_model /= amax(sum_model)

            sum_model_all.extend(sum_model)
            print('[%d] R^2: %.4f' % (
                ii,
                r_squared(loaded_data[ii][:, 1], sum_model),
            ))

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


"""
nmax = 9

states_vec = list(
    itertools.product(
        range(1, nmax + 1, 2),
        range(1, nmax + 1, 2),
    ))

"""
states_vec = [
    (1, 1),
    (1, 3),
    (1, 5),
    (3, 1),
    (5, 1),
]

N_states = len(states_vec)

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

sizes_vec = [(29.32, 5.43), (26.11, 6.42), (25.4, 8.05), (13.74, 13.37)]
hwhm_vec = [(3.3, 0.81), (3.34, 1.14), (2.9, 0.95), (2.17, 1.85)]

sys = system_data(m_e, m_h, eps_sol, T, size_d, 1, 1, eps_sol)

loaded_data = array([
    load_raw_PL_data('extra/data/extcharge/%s' % f, 0.11)
    for f in os.listdir('extra/data/extcharge')
])

concat_data = concatenate(loaded_data[:])

popt_list, pcov_list = [], []

pool = multiprocessing.Pool(multiprocessing.cpu_count())
"""
for ii, ((Lx, Ly), (hwhm_x, hwhm_y)) in enumerate(zip(sizes_vec, hwhm_vec)):
    popt, pcov = time_func(
        curve_fit,
        model_data(Lx, Ly, hwhm_x, hwhm_y, sys, pool),
        loaded_data[ii, :, 0],
        loaded_data[ii, :, 1],
        p0=(20e-3, -0.01),
        bounds=((0, -inf), (inf, inf)),
        method='trf',
    )

    popt_list.append(popt)
    pcov_list.append(pcov)

    print(popt)
    print(pcov)

print(popt_list)
print(pcov_list)
"""

popt, pcov = time_func(
    curve_fit,
    model_data_all(loaded_data, sizes_vec, hwhm_vec, sys, pool),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=(20e-3, -0.02, -0.01, -0.02, -0.01),
    bounds=((0, -inf, -inf, -inf, -inf), inf),
    method='trf',
)

print(popt)
print(pcov)

pool.terminate()

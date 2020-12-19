from common import *
import matplotlib.pyplot as plt
matplotlib.use('pdf')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)


def find_eps_sol(N_k, sys, be_cou, be_bnd):
    def find_f(eps_sol):
        sys.params.eps_sol = eps_sol
        return topo_be_p_cou(N_k, sys, be_bnd) - be_cou

    return root_scalar(
        find_f,
        bracket=(2, 10),
        method='brentq',
    ).root


be_bnd = 0.5

file_version = 'v5'

if file_version == 'v1':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.2, N_Q)
    be_func = topo_be_t_eff_cou_Q
elif file_version == 'v2':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.045, N_Q)
    be_func = topo_be_t_eff_cou_Q
elif file_version == 'v3':
    N_Q = 1 << 7
    Q_vec = linspace(0, 0.125, N_Q)
    be_func = topo_be_t_eff_cou_Q
elif file_version == 'v4':
    N_Q = 1 << 7
    Q_vec = linspace(0, 0.125, N_Q)
    be_func = topo_be_p_cou
elif file_version == 'v5':
    N_Q = 1 << 7
    Q_vec = linspace(0, 0.125, N_Q)
    be_func = topo_be_p_cou

be_arr = zeros((N_Q))


def compute_be(be_cou_func):
    for i in range(N_Q):
        print(
            '[%d/%d] Computing %.5f' % (
                i + 1,
                N_Q,
                Q_vec[i],
            ),
            flush=True,
        )

        if i >= 90:
            be_arr[i:] = float('nan')
            break
        else:
            be_arr[i] = time_func(
                be_cou_func,
                Q_vec[i],
                N_k,
                sys,
                be_bnd,
            )
            print('be: %.2f meV' % (be_arr[i] * 1e3))

    save(
        'extra/data/topo/%s_data_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        be_arr,
    )


#under_threshold = float('inf')
under_threshold = -float('inf')

try:
    be_arr[:] = load(
        'extra/data/topo/%s_data_%s.npy' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version))

    if (be_arr < under_threshold).any():
        compute_be(be_func)

except IOError as e:
    print('%s' % e, flush=True)

    compute_be(be_func)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

k_vec = linspace(-Q_vec[-1], Q_vec[-1], 1 << 8)
k_int_vec = linspace(0.0, Q_vec[-1], 1 << 8)
disp_vec = array([topo_eigenval_2d_v(0.5 * k, sys) for k in k_vec])
disp_int_vec = array(
    [amin([topo_disp_t_int(Q, k, sys) for k in k_int_vec]) for Q in Q_vec])
"""
be_arr -= topo_disp_t_shift(sys)
disp_vec -= topo_disp_t_shift(sys)
disp_int_vec -= topo_disp_t_shift(sys)
"""

ax[0].axhline(
    y=0,
    color='k',
    linestyle='-',
    linewidth=0.6,
)

ax[-1].plot(
    Q_vec,
    disp_int_vec,
    color='b',
    label=r'$\int d\theta E^c(\vec{Q}+\vec{k}) - E^v(\vec{Q}-\vec{k})$',
    linewidth=0.7,
)

ax[0].plot(
    k_vec,
    disp_vec[:, 2],
    color='g',
    label=r'$E^c(Q)$',
    linewidth=0.9,
)

ax[0].plot(
    k_vec,
    disp_vec[:, 0],
    color='c',
    label=r'$E^v(Q)$',
    linewidth=0.9,
)

ax[0].plot(
    -Q_vec,
    disp_int_vec,
    color='b',
    linewidth=0.9,
)

ax[0].plot(
    Q_vec,
    be_arr,
    'r-',
    label=r'$E_X$',
    linewidth=1.0,
)

ax[0].plot(
    -Q_vec,
    be_arr,
    'r-',
    linewidth=1.0,
)

ax[0].plot(
    [Q_vec[isnan(be_arr) == False][-1]],
    [be_arr[isnan(be_arr) == False][-1]],
    color='w',
    linestyle='',
    marker='o',
    markersize=4,
    markeredgecolor='r',
    markeredgewidth=0.7,
)

ax[0].plot(
    [-Q_vec[isnan(be_arr) == False][-1]],
    [be_arr[isnan(be_arr) == False][-1]],
    color='w',
    linestyle='',
    marker='o',
    markersize=4,
    markeredgecolor='r',
    markeredgewidth=0.7,
)

ax[0].set_xlim(-Q_vec[-1], Q_vec[-1])

ax[0].set_xlabel('Q (nm$^{-1}$)')
ax[0].set_ylabel('E (eV)')
ax[0].legend(
    loc='lower center',
    numpoints=3,
    prop={'size': 10},
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

#plt.show()

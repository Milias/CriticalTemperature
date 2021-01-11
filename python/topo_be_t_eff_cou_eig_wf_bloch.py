from common import *
import matplotlib.pyplot as plt
#matplotlib.use('pdf')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = (6.8, 5.3)

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

file_version = 'v3'

if file_version == 'v1':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.2, N_Q)
elif file_version == 'v2':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.045, N_Q)
elif file_version == 'v3':
    N_Q = 1 << 7
    Q_vec = linspace(0, 0.125, N_Q)

N_states_max = 6
N_Q_max = 128
k_max = 8.0
print('k_max: %f' % k_max)

eig_vec = full((N_Q_max, N_k), float('nan'))

try:
    eig_vec = load('extra/data/topo/%s_data_%s.npy' % (
        'topo_be_t_eff_cou_eig_wf',
        file_version,
    ))

except IOError as e:
    print('%s' % e, flush=True)

for ii, Q_val in enumerate(Q_vec[:N_Q_max]):
    if not isnan(eig_vec[ii, 0]):
        continue

    eig_arr = array(time_func(
        topo_t_eff_cou_eig,
        Q_val,
        k_max,
        N_k,
        sys,
    )).reshape(N_k + 1, N_k)

    eig_vec[ii] = eig_arr[1]

    save(
        'extra/data/topo/%s_data_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        eig_vec,
    )

    print('[%d/%d] (Q, E_Q): (%f, %f)' % (
        ii + 1,
        N_Q,
        Q_val,
        eig_arr[0, 0],
    ))

    del eig_arr

Q_idx = 30
Q_val = Q_vec[Q_idx]
phi_val = 0.25 * pi
k_vec = linspace(0, k_max, N_k)

wf_interp = interp1d(k_vec, eig_vec[Q_idx])

bloch_c_vec = array([[
    topo_bloch_cx_th(Q_val, phi_val, k, sys),
    topo_bloch_cy_th(Q_val, phi_val, k, sys),
    topo_bloch_cz_th(Q_val, phi_val, k, sys),
] for k in k_vec])

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

plot_labels = ['$\sigma^x(k)$', '$\sigma^y(k)$', '$\sigma^z(k)$']

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]

ax[0].axhline(
    y=0,
    color='k',
    linewidth=0.6,
)

ax[0].axvline(
    x=Q_val,
    color='m',
    linewidth=0.5,
)

for i in range(3):
    ax[0].plot(
        k_vec,
        bloch_c_vec[:, i] * (wf_interp(k_vec)**2),
        color=colors[i],
        linewidth=1.2,
        label=plot_labels[i],
    )

    ax[0].axhline(
        y=trapz(bloch_c_vec[:, i] * (wf_interp(k_vec)**2), k_vec),
        linewidth=0.7,
        color=colors[i],
    )

ax[0].set_xlabel('$k$ (nm$^{-1}$)')
ax[0].set_xlim(k_vec[0], 1.0)
#ax[0].set_ylim(-1, 1)

ax[0].legend(
    loc='upper right',
    prop={'size': 12},
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()

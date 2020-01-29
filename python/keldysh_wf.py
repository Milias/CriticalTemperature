from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]


def size_d_eff(sys):
    return sys.size_d * sys.eps_mat / sys.eps_r


def hn_r_pot_norm(x, sys):
    return -pi * (special.struve(0, 2 * x / size_d_eff(sys)) -
                  special.y0(2 * x / size_d_eff(sys)))


N_x, N_eps = 1 << 10, 5

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.22, 0.41, 294  # K

eps_vec = eps_sol / array([1e-3, 0.1, 0.2, 0.5, 1.0])

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_vec = [system_data(m_e, m_h, eps_sol, T, size_d, eps) for eps in eps_vec]

print(8 * pi**2 / sys_sol.c_aEM * sys_sol.c_hbarc / m_e / sys_sol.c_m_e)
print(8 * pi**2 / sys_sol.c_aEM * sys_sol.c_hbarc / m_h / sys_sol.c_m_e)
print(8 * pi**2 / sys_sol.c_aEM * sys_sol.c_hbarc / m_e / sys_sol.c_m_e / 1.37)
print(8 * pi**2 / sys_sol.c_aEM * sys_sol.c_hbarc / m_h / sys_sol.c_m_e / 1.37)
exit()

x_vec = logspace(log10(1e-3), log10(10), N_x)

hn_vec = array([hn_r_pot_norm(x_vec, sys) for sys in sys_vec]).reshape(
    (N_eps, N_x)).T

ke_vec = array([exciton_pot_ke_vec(x_vec, sys) for sys in sys_vec]).reshape(
    (N_eps, N_x)).T

load_data_from_file = True

if load_data_from_file == False:
    be_ke_args = [(
        exciton_be_ke,
        sys,
    ) for sys in sys_vec]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    be_ke_vec = array(time_func(
        pool.starmap,
        time_func,
        be_ke_args,
    ))

    print(be_ke_vec)

    wf_ke_args = [(
        exciton_wf_ke,
        be,
        x_vec[-1],
        N_x,
        sys,
    ) for be, sys in zip(be_ke_vec, sys_vec)]

    wf_ke_vec = array(time_func(
        pool.starmap,
        time_func,
        wf_ke_args,
    )).reshape((N_eps, N_x + 1, 3))

    pool.terminate()

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/keldysh/be_wf_ke_%s' % file_id,
        [
            be_ke_vec.flatten(),
        ],
        extra_data={
            'eps_vec': eps_vec.tolist(),
            'eps_sol': eps_sol,
            'size_d': size_d,
            'm_e': m_e,
            'm_h': m_h,
            'T': T,
        },
    )
else:
    file_id = 'pjp4LbKZTRGyLrQCPEuneQ'
    data = load_data('extra/keldysh/be_wf_ke_%s' % file_id, globals())

    eps_vec = array(eps_vec)
    N_eps = eps_vec.size

    be_ke_vec = array(data[0][:N_eps])

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    print(be_ke_vec)

    wf_ke_args = [(
        exciton_wf_ke,
        be,
        x_vec[-1],
        N_x,
        sys,
    ) for be, sys in zip(be_ke_vec, sys_vec)]

    wf_ke_vec = array(time_func(
        pool.starmap,
        time_func,
        wf_ke_args,
    )).reshape((N_eps, N_x + 1, 3))

    pool.terminate()

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, eps_vec.size)
]

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

    ax[0].plot(
        x_vec,
        hn_vec[:, n_eps],
        '-',
        color=c,
        label=r'$d^* / d$: $%d$' % (sys.eps_mat / sys.eps_r),
    )
    ax[0].plot(
        x_vec,
        ke_vec[:, n_eps] / energy_norm,
        '--',
        color=c,
    )

    ax[0].axvline(
        x=sys.exc_bohr_radius_mat(),
        linestyle='-',
        linewidth=0.7,
        color=c,
    )

    ax[1].plot(
        wf_ke_vec[n_eps, :, 2],
        wf_ke_vec[n_eps, :, 0] / 5e-9,
        '-',
        color=c,
    )

    ax[1].axvline(
        x=sys.exc_bohr_radius_mat(),
        linestyle='-',
        linewidth=0.7,
        color=c,
    )

ax[0].legend()

ax[0].set_xlim(0, x_vec[-1])
ax[0].set_ylim(-4, 0)

ax[0].set_xticks([])

ax[1].set_xlim(0, x_vec[-1])
ax[1].set_ylim(0, 0.8)

plt.tight_layout()

fig.subplots_adjust(hspace=0)

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'be_wf_v1')

plt.show()

from common import *

fig_size = tuple(array([6.8 * 2, 5.3 * 2]))

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

N_Q = 1 << 7
Q_vec = linspace(0, 0.125, N_Q)

try:
    os.mkdir('/storage/Reference/Work/University/PhD/TopoExciton/cou_mat')
except:
    pass


def cou_mat_savefig(ii, Q_vec):
    Q_val = Q_vec[ii]
    uQ_val = 1 - 1 / (1 + Q_val)

    result = array(time_func(
        topo_eff_cou_Q_mat,
        Q_val,
        N_k,
        sys,
    )).reshape(N_k, N_k)[::-1, ::-1]

    import matplotlib.pyplot as plt
    matplotlib.use('pdf')

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': False,
    })

    n_x, n_y = 1, 1
    fig = plt.figure(figsize=fig_size)
    ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

    plot_max = 2.0

    ax[0].imshow(
        result,
        cmap=cm.cividis,
        aspect='auto',
        extent=(0, 1, 0, 1),
        interpolation='none',
        vmin=-plot_max,
        vmax=plot_max,
        origin='lower',
    )

    ax[0].axvline(
        x=uQ_val,
        color='w',
        linewidth=0.8,
    )

    ax[0].axhline(
        y=uQ_val,
        color='w',
        linewidth=0.8,
    )

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(
        '/storage/Reference/Work/University/PhD/TopoExciton/cou_mat_2/%0d_%s_%s.pdf'
        % (
            ii,
            os.path.splitext(os.path.basename(__file__))[0],
            'v2',
        ),
        transparent=True,
        dpi=300,
    )

    plt.close()


pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

try:
    time_func(
        pool.map,
        functools.partial(cou_mat_savefig, Q_vec=Q_vec),
        range(Q_vec.size),
    )
except FileNotFoundError as e:
    print('Error: %s' % e)
    pool.close()
    exit()

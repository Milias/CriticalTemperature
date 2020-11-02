from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8 * 2, 5.3 * 2]))

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

Q_val = 0

#N_k = 1 << 6
uQ_val = 1 - 1 / (1 + Q_val)

result = array(time_func(
    topo_eff_cou_Q_mat,
    Q_val,
    N_k,
    sys,
)).reshape(N_k, N_k)[::-1, ::-1]

print(amax(result))
print(amin(result))
print(average(result))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

plot_max = 1.5

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
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v3'),
    transparent=True,
)

plt.show()

from common import *
import matplotlib.pyplot as plt

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


"""
eps_sol = find_eps_sol(N_k, sys, 193e-3, 1)
print(eps_sol)
"""

be_bnd = 5e-1

alpha_vec = linspace(0, 1, 1 << 6)
be_vec = array(topo_be_t_eff_cou_vec(alpha_vec, N_k, sys, be_bnd))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 1)
]

ax[0].plot(
    alpha_vec,
    be_vec * 1e3,
    color=colors[0],
    linestyle='-',
)

ax[0].set_xlim(alpha_vec[0], alpha_vec[-1])
ax[0].set_ylim(0, None)
ax[0].set_xlabel(r'$\alpha$')
ax[0].set_ylabel(r'$E_X$ (meV)')

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()

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
        return topo_be_t_cou(N_k, sys, be_bnd) - be_cou

    return root_scalar(
        find_f,
        bracket=(2, 10),
        method='brentq',
    ).root


"""
eps_sol = find_eps_sol(N_k, sys, 193e-3, 1)
print(eps_sol)
"""

be_bnd = 4e-2
exit()

be_cou = topo_be_t_eff_cou_cvcv(N_k, sys, 0)
print(be_cou)


z_vec = linspace(5e-5, be_bnd, 1 << 9)
det_vec = topo_det_t_eff_cou_cvcv_vec(z_vec, N_k, sys)
#print(det_vec)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

ax[0].plot(z_vec, det_vec, 'r.')
ax[0].set_ylim(-5e-5, 1e-4)

ax[0].axvline(
    x=be_cou,
    color='m',
    linewidth=0.9,
)

ax[0].axhline(
    y=0,
    color='k',
    linewidth=0.7,
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()

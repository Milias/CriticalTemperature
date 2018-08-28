from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

mu_a, mu_h, n_id, n_ex, n_sc = job_api.loaded_jobs

n, sys_iter = mu_a.args
sys = next(sys_iter)

#print(1e15 / 4 * 2 * pi * sys.c_hbar**2 * sys.c_light**2 / (sys.c_kB * sys.m_p))
#exit()

mu_a_arr = array(mu_a.result)

x = array(list(n))# * sys.lambda_th**-2

y, y2 = mu_a_arr[:, 0], mu_a_arr[:, 1]
y += array(mu_h.result)

n_id_arr, n_ex_arr, n_sc_arr = tuple([array(n.result) for n in job_api.loaded_jobs[-3:]])

#n_total = n_id_arr + n_ex_arr + n_sc_arr
#n_total = x
n_total = ones_like(x)

n_id_y = n_id_arr / n_total
n_ex_y = n_ex_arr / n_total
n_sc_y = n_sc_arr / n_total

ls_arr = array([1/ideal_2d_ls(n, sys) for n in n_id_arr])
#b_ex_energy = array([wf_2d_E_py(1/ls, sys) for ls in ls_arr])

plot_type = 'plot'

plt.autoscale(enable = True, axis = 'x', tight = True)
plt.tight_layout()
#getattr(plt, plot_type)(x, y , 'r-', label = r'$\mu_\gamma$')
getattr(plt, plot_type)(x, ls_arr, 'r-', label = r'$\lambda_s^{-1}$')
#getattr(plt, plot_type)(x, b_ex_energy, 'r--', label = r'$\chi_{ex}$')
getattr(plt, plot_type)(x, y2, 'b-', label = r'$a^{-1}$')
getattr(plt, plot_type)(x, n_ex_y + n_sc_y, 'm--', label = r'$n_{ex} + n_{sc}$')
getattr(plt, plot_type)(x, n_sc_y, 'm-', label = r'$n_{sc}$')
getattr(plt, plot_type)(x, n_ex_y, 'k-', label = r'$n_{ex}$')
getattr(plt, plot_type)(x, n_id_y, 'g-', label = r'$n_{id}$')
#plt.axis([x[0], x[-1], -1, 1])
plt.legend(loc = 0)
plt.show()


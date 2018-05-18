from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

mu_e, mu_h, n_id, n_ex, n_sc = job_api.loaded_jobs

mu_t, sys_iter = mu_e.args
chi_ex, mu_e, mu_h, sys_iter = n_ex.args
sys = next(sys_iter)

n_id_arr, n_ex_arr, n_sc_arr = [array(n.result) for n in (n_id, n_ex, n_sc)]

n_total = n_id_arr + n_ex_arr + n_sc_arr
#n_total = ones_like(n_total)

x = array(list(mu_t))

"""
plt.plot(x, [ideal_2d_mu(n, sys) + ideal_2d_mu_h(ideal_2d_mu(n, sys), sys) for n in n_id_arr], 'r-')
plt.plot(x, [analytic_2d_mu_ex(a, n, sys) for a, n in zip(x, n_ex_arr)], 'b-')
plt.plot(x, log(4 * pi * sys.m_pe * (1 + sys.m_pe) * n_ex_arr) - 8 / pi * x**2, 'g-')
plt.plot(x, - 8 / pi * x**2, 'g--')

plt.show()

exit()
"""

y = n_ex_arr / n_total
y2 = n_sc_arr / n_total
y3 = n_id_arr / n_total
y4 = (n_ex_arr + n_sc_arr) / n_total

plot_type = 'plot'

getattr(plt, plot_type)(x, y , 'r.-', label = 'Excitonic')
getattr(plt, plot_type)(x, y2, 'b.-', label = 'Scattering')
getattr(plt, plot_type)(x, y3, 'g.-', label = 'Ideal')
getattr(plt, plot_type)(x, y4, 'm.--', label = r'$n_{ex} + n_{sc}$')
plt.legend(loc = 0)
plt.tight_layout()
plt.autoscale(enable = True, axis = 'x', tight = True)
#plt.axis([x[0], x[-1], -0.3, 0.6])

plt.show()


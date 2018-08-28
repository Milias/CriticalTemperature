from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

mu_ls, mu_h, chi_ex, n_id, n_ex, n_sc = job_api.loaded_jobs

mu_ls_arr = array(mu_ls.result)
sys = next(n_ex.args[-1])

n_id_arr, n_ex_arr, n_sc_arr = [array(n.result) for n in (n_id, n_ex, n_sc)]

x = array(list(mu_ls.args[0]))

#n_total = n_id_arr + n_ex_arr + n_sc_arr
#n_total = ones_like(x)
n_total = x

y = n_ex_arr / n_total
y2 = n_sc_arr / n_total
y3 = n_id_arr / n_total
y4 = (n_ex_arr + n_sc_arr) / n_total

plot_type = 'semilogx'

getattr(plt, plot_type)(x, y , 'r-', label = 'Excitonic')
getattr(plt, plot_type)(x, y2, 'b-', label = 'Scattering')
getattr(plt, plot_type)(x, y3, 'g-', label = 'Ideal')
getattr(plt, plot_type)(x, y4, 'm--', label = r'$n_{ex} + n_{sc}$')

plt.legend(loc = 0)
plt.tight_layout()
plt.autoscale(enable = True, axis = 'x', tight = True)

#plt.axis([x[0], x[-1], -0.3, 0.6])

plt.show()


from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

mu_a, mu_h, n_id, n_ex, n_sc = job_api.loaded_jobs

n, sys_iter = mu_a.args
sys = next(sys_iter)

mu_a_arr = array(mu_a.result)
nan_count = count_nonzero(~isnan(mu_a_arr[:, 1]))

x = array(list(n))[:nan_count] * sys.lambda_th**-2

y, y2 = mu_a_arr[:nan_count, 0], mu_a_arr[:nan_count, 1]

n_id_arr, n_ex_arr, n_sc_arr = tuple([array(n.result)[:nan_count] for n in (n_id, n_ex, n_sc)])

n_total = n_id_arr + n_ex_arr + n_sc_arr

n_id_y = n_id_arr / n_total
n_ex_y = n_ex_arr / n_total
n_sc_y = n_sc_arr / n_total

plot_type = 'semilogx'

#getattr(plt, plot_type)(x, y , 'r-', label = 'Chemical potential')
getattr(plt, plot_type)(x, y2, 'b-', label = 'Scattering length')
getattr(plt, plot_type)(x, n_ex_y + n_sc_y, 'm-', label = r'$n_{ex} + n_{sc}$')
#getattr(plt, plot_type)(x, n_id_y, 'g-', label = r'$n_{id}$')
plt.legend(loc = 0)
plt.show()


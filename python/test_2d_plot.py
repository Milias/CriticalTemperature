from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

n_id, n_ex, n_sc = job_api.loaded_jobs

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

n, a, sys = n_ex.args
sys = system_data(m_e, m_h, eps_r, T)

n_id_arr, n_ex_arr, n_sc_arr = array(n_id.result), array(n_ex.result), array(n_sc.result)

n_total = (n_id_arr + n_ex_arr + n_sc_arr)

x = array(list(a))

plt.plot(x, [ideal_2d_mu(n, sys) + ideal_2d_mu_h(ideal_2d_mu(n, sys), sys) for n in n_id_arr], 'r-')
plt.plot(x, [analytic_2d_mu_ex(a, n, sys) for a, n in zip(x, n_ex_arr)], 'b-')
plt.plot(x, log(4 * pi * sys.m_pe * (1 + sys.m_pe) * n_ex_arr) - 8 / pi * x**2, 'g-')
plt.plot(x, - 8 / pi * x**2, 'g--')

plt.show()

exit()

y = n_ex_arr / n_total
y2 = n_sc_arr / n_total
#y3 = n_id_arr / n_total
y4 = (n_ex_arr + n_sc_arr) / n_total

plot_type = 'plot'

getattr(plt, plot_type)(x, y , 'r-', label = 'Excitonic')
getattr(plt, plot_type)(x, y2, 'b-', label = 'Scattering')
#getattr(plt, plot_type)(x, y3, 'g-', label = 'Ideal')
getattr(plt, plot_type)(x, y4, 'm-', label = r'$n_{ex} + n_{sc}$')
plt.legend(loc = 0)
plt.show()


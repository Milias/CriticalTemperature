from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

mu_a, = job_api.loaded_jobs

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

n, sys_iter = mu_a.args
sys = next(sys_iter)

y_arr = array(mu_a.result)
nan_count = count_nonzero(~isnan(y_arr[:, 1]))

x = array(list(n))[:nan_count] * sys.lambda_th**-2

y, y2 = y_arr[:nan_count, 0], y_arr[:nan_count, 1]

plot_type = 'semilogx'

#getattr(plt, plot_type)(x, y , 'r-', label = 'Chemical potential')
getattr(plt, plot_type)(x, y2, 'b-', label = 'Scattering length')
plt.legend(loc = 0)
plt.show()


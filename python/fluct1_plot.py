from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

n_ex, = job_api.loaded_jobs

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

mu_e, mu_h, sys = tuple([n_ex.args[i].__reduce__()[1][0] for i in [1, 2, 3]])

x = array([x for x in n_ex.args[0]])
y = n_ex.result

plt.plot(x, y, 'r.-')
plt.axvline(x = fluct_pp0_a(mu_e, mu_h, sys), color = 'g')
plt.axvline(x = fluct_ac(0, mu_e, mu_h, sys), color = 'k')
plt.show()


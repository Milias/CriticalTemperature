from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

green, = job_api.loaded_jobs

W, K1_iter, K2_iter, mu_e_iter, mu_h_iter, sys_iter = green.args
K1, K2, mu_e, mu_h, sys = [next(it) for it in (
  K1_iter, K2_iter, mu_e_iter, mu_h_iter, sys_iter
)]

N_total = sys_iter.__length_hint__()
N_k = 1 << 0
N_w = 1 << 12

green_arr, = [array(green.result) for n in job_api.loaded_jobs]

plt.plot(W, green_arr[:, 0], 'r-')
plt.plot(W, green_arr[:, 1], 'b-')

plt.show()


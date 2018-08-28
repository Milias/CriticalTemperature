from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

green, = job_api.loaded_jobs

w_iter, k, mu_e_iter, mu_h_iter, sys_iter = green.args
w, mu_e, mu_h, sys = [next(it) for it in (w_iter, mu_e_iter, mu_h_iter, sys_iter)]

N = k.size
th = linspace(0, 2*pi, N)

green_arr, = [-array(green.result) for n in job_api.loaded_jobs]

zr, zi = green_arr[:,0], green_arr[:,1]

plt.plot(th, zr, 'r-')
plt.plot(th, zi, 'b-')

plt.show()


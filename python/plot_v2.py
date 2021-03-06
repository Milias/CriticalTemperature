from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

job_api.load_batch()

(mu, init_mu, mu_f, ideal_mu) = job_api.loaded_jobs

x = array([x for x in mu.args[0]])

# Get repeated object
sys = mu.args[1].__reduce__()[1][0]

x *= sys.lambda_th**-3

y = zeros_like(x)
y2 = zeros_like(x)
y3 = zeros_like(x)
y4 = zeros_like(x)

result = array(mu.result)
y = result[:, 0]
y2 = result[:, 1]
"""
mu_f_result = array(mu_f.result)
y3 = mu_f_result[:, 0]
y4 = 1/mu_f_result[:, 1]
"""

#plot_type = 'plot'
plot_type = 'semilogx'

axplots = []
fig, axarr = plt.subplots(1, 1, sharex = True, figsize = (16, 5), dpi = 96)
#fig.subplots_adjust(hspace=0)
fig.tight_layout()

axplots.append(getattr(axarr, plot_type)(x, real(y), 'r-', marker = '.'))

axarr.autoscale(enable = True, axis = 'x', tight = True)

axarr.plot(x, imag(y), 'r--', marker = '.')
axarr.plot(x, real(y2), 'b-', marker = '.')
axarr.plot(x, imag(y2), 'b--', marker = '.')
axarr.plot(x, y3, 'g-', marker = '.')
axarr.plot(x, y4, 'm-', marker = '.')

#axarr.axhline(y = z0, linestyle = '-', color = 'g')
#axarr.axvline(x = z0, linestyle = '--', color = 'g', linewidth = 1)

if nanmax(y) >= 0 and nanmin(y) <= 0:
  axarr.axhline(y = 0, linestyle = '-', color = 'k', linewidth = 0.5)

if x[0] <= 0 and x[-1] >= 0:
  axarr.axvline(x = 0, linestyle = '-', color = 'k', linewidth = 0.5)

axarr.set_ylim(-100, 100)
axarr.set_xlim(x[0], x[-1])
#axarr.set_yscale('symlog')

#axarr.set_xticks([0.0], minor = False)
#axarr.set_yticks([0.0], minor = False)
#axarr.grid(color = 'k', linestyle = '-', linewidth = 0.5)

t_now = 1e6 * time.time()
fig.savefig('bin/plots/figure_%d.eps' % t_now)
fig.savefig('bin/plots/figure_%d.png' % t_now)

plt.show()


from common import *

job_api = JobAPI()

job_filename = job_api.getLastJob()['output_file']

print('Loading from file: %s' % job_filename)
job = job_api.loadData(job_filename)

x = job['args'][0]

# Get repeated object
sys = job['args'][1].__reduce__()[1][0]

x *= sys.lambda_th**-3

y = zeros_like(x)
y2 = zeros_like(x)
y3 = zeros_like(x)
y4 = zeros_like(x)

result = array(job['result'])
y = result[:, 0]
y2 = result[:, 1]

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

#axarr.set_ylim(-25, 3)
axarr.set_xlim(x[0], x[-1])
#axarr.set_yscale('symlog')

#axarr.set_xticks([0.0], minor = False)
#axarr.set_yticks([0.0], minor = False)
#axarr.grid(color = 'k', linestyle = '-', linewidth = 0.5)

t_now = 1e6 * time.time()
fig.savefig('bin/plots/figure_%d.eps' % t_now)
fig.savefig('bin/plots/figure_%d.png' % t_now)

plt.show()


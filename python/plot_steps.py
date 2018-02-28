from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

mu_steps, = job_api.loaded_jobs

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

def plot_steps(axarr, y):
  axarr[0].plot(y[::4], y[2::4], '.--')
  axarr[0].plot([y[0]], [y[2]], 'kx')
  axarr[0].plot([y[-4]], [y[-2]], 'ko')

  axarr[0].set_ylabel('Scattering length')
  axarr[0].set_xlabel('Chemical potential')

  axarr[1].axhline(y = 0, color = 'k', linestyle = '-')
  axarr[1].axvline(x = 0, color = 'k', linestyle = '-')

  axarr[1].plot(y[1::4], y[3::4], '.--')
  axarr[1].plot([y[1]], [y[3]], 'kx')
  axarr[1].plot([y[-3]], [y[-1]], 'ko')

  axarr[1].set_xlabel('Equation of state')
  axarr[1].set_ylabel('Self-consistency condition')

fig, axarr = plt.subplots(1, 2, figsize = (18, 6), dpi = 96)

for y in mu_steps.result:
  plot_steps(axarr, y)

plt.show()


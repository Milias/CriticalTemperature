from common import *

def analytic_plots(N = 1<<8):
  m_1, m_2 = 0.28 * m_electron, 0.59 * m_electron # eV
  m_r = 1.0 / (1.0 / m_1 + 1.0 / m_2) # eV

  T = 300 # K
  beta = 1 / (k_B * T) # eV^-1
  lambda_th = c * hbar * sqrt(2 * pi * beta / m_r) # m
  energy_th = 1 / ( 4 * pi * beta )
  eps_r, e_ratio = 6.56, m_r / energy_th
  m_ratio_e = m_1 / m_r
  m_ratio_h = m_2 / m_r

  print('%f nm, %f meV' % (lambda_th * 1e9, energy_th * 1e3))

  x = linspace(3e23, 3e24, N) * lambda_th**3

  t0 = time.time()

  mu_arr = array(parallelTable(analytic_mu, x, itertools.repeat(m_ratio_e, N), itertools.repeat(m_ratio_h, N), itertools.repeat(eps_r, N), itertools.repeat(e_ratio, N)))

  ideal_mu_arr_e = array(parallelTable(ideal_mu, x, itertools.repeat(m_ratio_e, N)))
  ideal_mu_arr_h = array(parallelTable(ideal_mu, x, itertools.repeat(m_ratio_h, N)))

  mu_total = mu_arr[:, 0] + mu_arr[:, 1]
  ideal_mu_total = ideal_mu_arr_e + ideal_mu_arr_h

  y_id = array(parallelTable(analytic_n_id, mu_arr[:,0], itertools.repeat(m_ratio_e, N))) + array(parallelTable(analytic_n_id, mu_arr[:,1], itertools.repeat(m_ratio_h, N)))

  y_ex = array(parallelTable(analytic_n_ex, mu_total, itertools.repeat(m_ratio_e + m_ratio_h, N), mu_arr[:, 2]))
  y_sc = array(parallelTable(analytic_n_sc, mu_total, itertools.repeat(m_ratio_e + m_ratio_h, N), mu_arr[:, 2]))
  y_ex_norm = y_ex / (y_id + y_sc + y_ex)
  y = (y_sc + y_ex) / (y_id + y_sc + y_ex)

  dt = time.time() - t0

  x *= lambda_th**-3

  mu_arr[:, 0:2] *= energy_th
  ideal_mu_arr_e[:] *= energy_th
  ideal_mu_arr_h[:] *= energy_th

  xi_vline = argmin((mu_arr[:, 2])**2)
  x_vline = x[xi_vline]

  y_ex_norm[xi_vline] = float('nan')
  mu_arr[xi_vline, 2] = float('nan')

  plot_type = 'plot'

  axplots = []
  fig, axarr = plt.subplots(3, 1, sharex = True, figsize = (8, 12), dpi = 96)
  fig.subplots_adjust(hspace=0)

  axplots.append(getattr(axarr[0], plot_type)(x, y, 'r-', label = r'$(n_{ex} + n_{sc})/n$'))
  axarr[0].autoscale(enable = True, axis = 'x', tight = True)
  axarr[0].plot(x, y_ex_norm, 'm--', label = '$n_{ex}/n$')
  axarr[0].set_ylabel('Density contributions, T = %.0f K' % T)
  axarr[0].legend(loc = 0)
  axarr[0].axvline(x = x_vline, linestyle = '-', color = 'g')

  axplots.append(getattr(axarr[1], plot_type)(x, mu_arr[:, 0], 'r-', label = r'$\mu_e$'))
  axarr[1].autoscale(enable = True, axis = 'x', tight = True)
  axarr[1].plot(x, mu_arr[:, 1], 'b-', label = r'$\mu_h$')
  axarr[1].plot(x, ideal_mu_arr_e, 'r--', label = r'$\mu_{e, id}$')
  axarr[1].plot(x, ideal_mu_arr_h, 'b--', label = r'$\mu_{h, id}$')
  axarr[1].set_ylabel(r'Chemical potential (eV)')
  axarr[1].legend(loc = 0)
  axarr[1].axvline(x = x_vline, linestyle = '-', color = 'g')

  axplots.append(getattr(axarr[2], plot_type)(x, 1/mu_arr[:, 2], 'g-'))
  axarr[2].autoscale(enable = True, axis = 'x', tight = True)
  axarr[2].set_xlabel(r'$n$ (m$^{-3}$)')
  axarr[2].set_ylabel(r'Scattering length --- $a/\Lambda_{th}$')
  axarr[2].set_ylim(-50, 50)
  axarr[2].axvline(x = x_vline, linestyle = '-', color = 'g')

  fig.savefig('bin/plots/analytic_n_ex_sc.eps')

  print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
  plt.show()


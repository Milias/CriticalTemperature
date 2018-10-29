from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

green, = job_api.loaded_jobs

W, K, mu_e_iter, mu_h_iter, v_1_iter, sys_iter, delta_iter = green.args
mu_e, mu_h, v_1, sys = [next(it) for it in (mu_e_iter, mu_h_iter, v_1_iter, sys_iter)]

W, K = array(list(W)), array(list(K))
N = int(sqrt(W.size))

green_arr, = [array(green.result) for n in job_api.loaded_jobs]

W, K = W.reshape((N, N)), K.reshape((N, N))
green_arr = green_arr.reshape((N, N, 2))
zr, zi = green_arr[:,:,0], green_arr[:,:,1]

print((amin(zi), amax(zi)))
p = 1e-1

#zr = clip(zr, -0.2, 0.2)[::-1, :]
zi = clip(zi, p * amin(zi), p * amax(zi))[::-1, :]
kvec = linspace(amin(K), amax(K), 4 * N + 1)
wvec = linspace(0, amax(W), 4 * N + 1)

if v_1 > 0:
  kmax = plasmon_kmax(mu_e, mu_h, v_1, sys)
  wmax = plasmon_wmax(kmax, mu_e, sys)

  print('kmax: %f, wmax: %f' % (kmax, wmax))

  wvec1 = sqrt((mu_e + mu_h) / (2 * pi * sys.eps_r * v_1) * abs(kvec))
  wvec2 = -wvec1

  wvec1b = sqrt(
    abs(kvec) * (mu_e + mu_h) / (4 * pi * sys.eps_r * v_1)
    * (
      1 + sqrt(
        1
        + 24 * pi * sys.eps_r * (sys.m_pe * mu_e**2 + sys.m_ph * mu_h**2) * abs(kvec) * v_1 / (mu_e + mu_h)**2
      )
    )
  )
  wvec2b = - wvec1b

  wplvec1 = array([plasmon_disp(abs(k), mu_e, mu_h, v_1, sys) for k in kvec])
  wplvec2 = -wplvec1

  print('end')

  plt.plot(kvec, wvec1, 'b--')
  plt.plot(kvec, wvec2, 'b--')

  plt.plot(kvec, wvec1b, 'k--')
  plt.plot(kvec, wvec2b, 'k--')

  plt.plot(kvec, wplvec1, 'k-')
  plt.plot(kvec, wplvec2, 'w-')

  plt.plot([0, kmax, kmax, 0, 0], [0, 0, wmax, wmax, 0], 'k-')

wvec1c = sys.m_pe * kvec**2 - 2 * sqrt(sys.m_pe * mu_e) * abs(kvec)
wvec2c = sys.m_ph * kvec**2 - 2 * sqrt(sys.m_ph * mu_h) * abs(kvec)

wvec3c = sys.m_pe * kvec**2 + 2 * sqrt(sys.m_pe * mu_e) * abs(kvec)
wvec4c = sys.m_ph * kvec**2 + 2 * sqrt(sys.m_ph * mu_h) * abs(kvec)

plt.imshow(
  zi,
  cmap = cm.hot,
  aspect = 'auto',
  extent = (
    amin(K),
    amax(K),
    amin(W),
    amax(W)
  ),
  norm = SymLogNorm(linthresh = 1e-4 * amax(zi))
)

plt.plot(kvec, wvec1c, 'g-')
plt.plot(kvec, wvec2c, 'g--')
plt.plot(kvec, -wvec1c, 'g-')
plt.plot(kvec, -wvec2c, 'g--')

plt.plot(kvec, wvec3c, 'm-')
plt.plot(kvec, wvec4c, 'm--')
plt.plot(kvec, -wvec3c, 'm-')
plt.plot(kvec, -wvec4c, 'm--')

plt.axis([
  amin(K),
  amax(K),
  amin(W),
  amax(W)
])

plt.colorbar()

"""
fig, axarr = plt.subplots(ncols = 2, sharey = True)
ax = [None for ax in axarr]

ax[0] = axarr[0].imshow(zr, cmap=cm.hot)
ax[1] = axarr[1].imshow(zi, cmap=cm.hot)

fig.tight_layout()
fig.subplots_adjust(right = 0.6)
cbar_ax = fig.add_axes([0.7, 0.15, 0.05, 0.7])
fig.colorbar(ax[1], cax=cbar_ax)
"""

plt.show()


from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

green, = job_api.loaded_jobs

W, K, mu_e_iter, mu_h_iter, sys_iter = green.args
mu_e, mu_h, sys = [next(it) for it in (mu_e_iter, mu_h_iter, sys_iter)]

N = int(sqrt(W.size))

green_arr, = [array(green.result) for n in job_api.loaded_jobs]

W, K = W.reshape((N, N)), K.reshape((N, N))
green_arr = green_arr.reshape((N, N, 2))
zr, zi = green_arr[:,:,0], green_arr[:,:,1]

print((amin(zi), amax(zi)))
p = 1e-3

#zr = clip(zr, -0.2, 0.2)[::-1, :]
zi = clip(zi, p * amin(zi), p * amax(zi))[::-1, :]

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
  norm = SymLogNorm(linthresh = p)
)

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


from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<4
N = 1<<8

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 30 # K
sys = system_data(m_e, m_h, eps_r, T)

print((sys.m_pe, sys.m_ph))

chi_ex = -30
mu_e = ideal_2d_mu_v(-45, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)
mu_t = mu_e + mu_h

print((mu_e, mu_h, mu_t))

epsj = 1e-3j
vw = linspace(0, 60, N)
vk = linspace(-15, 15, N)

w, k = meshgrid(vw, vk, indexing = 'xy')

I2_vals = zeros_like(w)
T2B_vals = zeros_like(w)

T2B_vals = array([cmath.log(cmath.sqrt(chi_ex / (x + epsj - y**2 / sys.m_sigma + mu_t))) for y, x in zip(k.ravel(), w.ravel())]).reshape(w.shape)
I2_vals = array([fluct_2d_I2(x, y**2, mu_e, mu_h, sys) for y, x in zip(k.ravel(), w.ravel())]).reshape(w.shape)

z1 = 1 / (w + epsj - k**2 / sys.m_sigma + mu_t) / (T2B_vals)
z = 1 / (w + epsj - k**2 / sys.m_sigma + mu_t) / (T2B_vals + I2_vals)

zr = clip(imag(z1).T, -1, 1)[::-1, :]
zi = clip(imag(z).T, -1, 1)[::-1, :]

fig, axarr = plt.subplots(ncols = 2, sharey=True)
ax = [None for ax in axarr]

ax[0] = axarr[0].imshow(zr, extent=(vk[0], vk[-1], vw[0], vw[-1]), cmap=cm.hot)
ax[1] = axarr[1].imshow(zi, extent=(vk[0], vk[-1], vw[0], vw[-1]), cmap=cm.hot)

fig.tight_layout()
fig.subplots_adjust(right=0.6)
cbar_ax = fig.add_axes([0.6, 0.15, 0.05, 0.7])
fig.colorbar(ax[1], cax=cbar_ax)

plt.show()

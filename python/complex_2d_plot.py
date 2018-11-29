from common import *

N_z = 1 << 8 + 1

z0_real, z0_imag = 3, 3
z_real_vec, z_imag_vec = linspace(-z0_real, z0_real, N_z), linspace(
    -z0_imag, z0_imag, N_z)

z_real_arr, z_imag_arr = meshgrid(z_real_vec, z_imag_vec)

z_arr = (z_real_arr + 1j * z_imag_arr).T

r, ph = abs(z_arr), angle(z_arr)

h = 0.5 + 0.5 * ph / pi
s = 0.9 * ones_like(r)
v = r / (1.0 + r)

hsv = array([h, s, v]).T

colors = matplotlib.colors.hsv_to_rgb(hsv)

plt.imshow(
    colors,
    extent=(-z0_real, z0_real, -z0_imag, z0_imag),
)

#plt.contour(z_real_arr.T, z_imag_arr.T, r, 16, cmap=cm.cool)

plt.savefig('plots/complex_map.eps')
plt.savefig('plots/complex_map.pdf')
plt.savefig('plots/complex_map.png')

plt.show()

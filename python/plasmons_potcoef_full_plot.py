from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

matrix, = job_api.loaded_jobs

wkwk_iter = list(matrix.args[1])
z_list = list(set(matrix.args[4]))
mu_e, mu_h, v_1, sys = [next(matrix.args[it]) for it in (5, 6, 7, 8)]

N_w, N_k, N_z = 1<<0, 1<<10, 1<<5
N_total = N_w**2 * N_k**2 * N_z

matrix_arr = array(matrix.result).reshape((N_z, N_w*N_k, N_w*N_k, 2))
matrix_complex_arr = matrix_arr[:, :, :, 0] + 1j * matrix_arr[:, :, :, 1]

"""
set_printoptions(threshold=nan)
print(green_arr)
#print(array(list(matrix[0].args[1])))
print(matrix_arr[0][0, :, :, 0])
print(matrix_arr[0][0, :, :, 1])
print(matrix_complex_arr[0][0])
exit()
"""

#print(matrix_complex_arr[0].shape)
#print(amin(matrix_arr[0][0,:,:,0]), amax(matrix_arr[0][0,:,:,0]))
#print(amin(matrix_arr[0][0,:,:,1]), amax(matrix_arr[0][0,:,:,1]))

"""
plt.plot(diag(matrix_arr[:,:,0], k = 0), 'r.-')
plt.plot(diag(matrix_arr[:,:,0], k = 1), 'b.-')
plt.plot(diag(matrix_arr[:,:,0], k = -1), 'g.-')
plt.show()
exit()
"""

#"""
p = 1
z_id = 0
mat = matrix_arr[z_id,:,:,0]
matmin, matmax = amin(mat), amax(mat)

zr = clip(
  mat,
  p * matmin,
  p * matmax
)

print(zr)

plt.imshow(
  zr,
  cmap = cm.hot,
  aspect = 'auto',
  norm = LogNorm()
)

plt.colorbar()
plt.tight_layout()

plt.show()
exit()
#"""

print('Computing eigenvalues')
"""
eig_vals = zeros((N_w, N_w, N_k))
for i in range(N_w):
  for j in range(1, N_w):
    print((i,j))
    eig_vals[i,j] = scipy.linalg.eigvals(matrix_complex_arr[i*N_k:(i+1)*N_k,j*N_k:(j+1)*N_k])
    plt.plot(eig_vals[i,j], '.-', label = '(%d,%d)' % (i,j))

plt.legend(loc = 0)
plt.plot()
exit()
"""

z = sort(-array(z_list))

"""
eig_vals = zeros((N_z, N_k*N_w), dtype = complex)
for i in range(N_z):
  eig_vals[i] = scipy.linalg.eigvals(matrix_complex_arr[i])
  plt.plot(real(eig_vals[i]), '.-', label = 'z: %f, real' % z[i])
  #plt.plot(imag(eig_vals[i]), '.--', label = 'z: %f, imag' % z[i])

plt.legend(loc = 0)
plt.show()
exit()
"""

#eig_vals = scipy.linalg.eigvals(matrix_complex_arr, check_finite = False)
#plt.plot(real(eig_vals), 'r.-')
#plt.plot(real(eig_vals), imag(eig_vals), 'o')
#plt.show()

det_vals = array([scipy.linalg.det(matrix_complex_arr[m,:,:]) for m in range(N_z)])

print(z)
print(det_vals)

plt.semilogx(z, real(det_vals), 'r.-')
plt.semilogx(z, imag(det_vals), 'b.--')

#plt.axis([amin(z), amax(z), -100, 100])
plt.show()


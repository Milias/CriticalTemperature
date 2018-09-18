from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

green = job_api.loaded_jobs[0]
matrix = job_api.loaded_jobs[1:]

wkk_iter, mu_e_iter, mu_h_iter, sys_iter = green.args
mu_e, mu_h, sys = [next(it) for it in (mu_e_iter, mu_h_iter, sys_iter)]

z_list = list(set(matrix[0].args[4]))
N_total = wkk_iter.__length_hint__
N_w, N_k, N_z = wkk_iter.N_w, wkk_iter.N_k, len(z_list)

green_arr = array(green.result)
matrix_arr = [array(n.result) for n in matrix]

green_arr = green_arr.reshape((N_w, N_k * (N_k + 1) // 2, 2))
green_complex_arr = green_arr[:, :, 0] + 1j * green_arr[:, :, 1]

matrix_arr = [m.reshape((N_z, N_w*N_k, N_w*N_k, 4)) for m in matrix_arr]

matrix_complex_arr_e = [m[:, :, :, 0] + 1j * m[:, :, :, 1] for m in matrix_arr]
matrix_complex_arr_h = [m[:, :, :, 2] + 1j * m[:, :, :, 3] for m in matrix_arr]

"""
set_printoptions(threshold=nan)
print(green_arr)
#print(array(list(matrix[0].args[1])))
print(matrix_arr[0][0, :, :, 0])
print(matrix_arr[0][0, :, :, 1])
print(matrix_complex_arr_e[0][0])
exit()
"""

print(matrix_complex_arr_e[0].shape)
print(amin(matrix_arr[0][0,:,:,0]), amax(matrix_arr[0][0,:,:,0]))
print(amin(matrix_arr[0][0,:,:,1]), amax(matrix_arr[0][0,:,:,1]))

"""
plt.plot(diag(matrix_arr[:,:,0], k = 0), 'r.-')
plt.plot(diag(matrix_arr[:,:,0], k = 1), 'b.-')
plt.plot(diag(matrix_arr[:,:,0], k = -1), 'g.-')
plt.show()
exit()
"""

"""
p = 1

zr = clip(
  matrix_arr[0][0,:,:,1],
  p * amin(matrix_arr[0][0,:,:,1]),
  p * amax(matrix_arr[0][0,:,:,1])
)

plt.imshow(
  zr,
  cmap = cm.hot,
  aspect = 'auto',
  #norm = LogNorm()
)

plt.colorbar()

plt.show()
exit()
"""

print('Computing eigenvalues')
"""
eig_vals = zeros((N_w, N_w, N_k))
for i in range(N_w):
  for j in range(1, N_w):
    print((i,j))
    eig_vals[i,j] = scipy.linalg.eigvals(matrix_complex_arr_e[i*N_k:(i+1)*N_k,j*N_k:(j+1)*N_k])
    plt.plot(eig_vals[i,j], '.-', label = '(%d,%d)' % (i,j))

plt.legend(loc = 0)
"""

#eig_vals = scipy.linalg.eigvals(matrix_complex_arr_e, check_finite = False)
#plt.plot(real(eig_vals), 'r.-')
#plt.plot(real(eig_vals), imag(eig_vals), 'o')
#plt.show()

det_vals = array([scipy.linalg.det(matrix_complex_arr_e[0][m,:,:]) for m in range(N_z)])
z = sort(-imag(array(z_list)))

print(z)
print(det_vals)

plt.semilogx(z, real(det_vals), 'r.-')
plt.semilogx(z, imag(det_vals), 'b.--')

#plt.axis([amin(z), amax(z), -1, 1])
plt.show()


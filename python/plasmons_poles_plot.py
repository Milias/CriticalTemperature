from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)
job_api.load_batch()

matrix, = job_api.loaded_jobs

wkwk_iter = list(matrix.args[1])
z_list = list(set(matrix.args[4]))
mu_e, mu_h, v_1, sys = [next(matrix.args[it]) for it in (5, 6, 7, 8)]

N_w, N_k, N_z = 1 << 0, 1 << 10, 1 << 5
N_total = N_w**2 * N_k**2 * N_z

matrix_arr = array(matrix.result).reshape((N_z, N_w * N_k, N_w * N_k, 2))
matrix_complex_arr = matrix_arr[:, :, :, 0] + 1j * matrix_arr[:, :, :, 1]

print(matrix_complex_arr[0].shape)

print('Starting computation')
c_det = plasmon_sysmat_det(matrix_complex_arr[0].flatten(),
                           matrix_complex_arr[0].shape)
s_det = 0  #scipy.linalg.det(matrix_complex_arr[0])

print((c_det, s_det))

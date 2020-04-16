import json
from numpy import array, loadtxt


def load_data(filename, extra_dict={}):
    try:
        exported_data = loadtxt('%s.csv.gz' % filename, delimiter=',').T
    except:
        exported_data = loadtxt('%s.csv' % filename, delimiter=',').T

    try:
        with open('%s.json' % filename, 'r') as fp:
            data_file = json.load(fp)
            extra_dict.update(data_file)
    except Exception as exc:
        print('load_data: %s' % exc)

    return exported_data


def load_PL(path, file_id):
    extra_dict = {}
    data = load_data(path + ('/cm_be_%s' % file_id), extra_dict)
    data = data.reshape((
        len(extra_dict['states_vec']),
        len(extra_dict['E_vec']),
        2,
    ))

    extra_dict_fit = {}
    data_at_fit = load_data(path + ('/cm_be_fit_%s' % file_id), extra_dict_fit)
    data_at_fit = data_at_fit.reshape((
        len(extra_dict_fit['states_vec']),
        len(extra_dict_fit['E_vec']),
        2,
    ))

    extra_dict['E_vec'] = array(extra_dict['E_vec'])
    extra_dict_fit['E_vec'] = array(extra_dict_fit['E_vec'])

    return (
        extra_dict['E_vec'],
        data,
        extra_dict_fit['E_vec'],
        data_at_fit,
        extra_dict,
        extra_dict_fit,
    )


"""
Always in this order:

(Lx   , Ly  ),
(29.32, 5.43),
(26.11, 6.42),
(25.40, 8.05),
(13.74, 13.37),

5 states, no BE distribution, 4 gammas.

file_id_list = [
    'X9A0Pvg0RL-cUoKemSowqw',
    'fMYnAIkNST-Sz0Y-BhbQwQ',
    'M0imMVJeTDWHWfGd_1C4ZA',
    'Ib0gxfBqS2uiOvsR_hnCzg',
]

5 states, BE distribution, 4 gammas.

file_id_list = [
    'ZrY6wM9GQ4ayN0jioNaWCQ',
    'Bw3fl1knTgKNeNuosNFOzQ',
    '2Xl6sk8kSZCaS4wQU9X4Gg',
    'NlYAify5Rwijs4yAKsMwqg',
]

5 states, BE distribution, 1 gamma.

file_id_list = [
    'sykV12o6Ty-UgQ0gdf5NbQ',
    'eBFHb6IwSxCw-fP61faLDA',
    '9RExzuAZSJatRBuTiOl1fQ',
    'o_ETwXT0QjWPyLfsFtZf1Q',
]

Up to n=9, BE, 1 gamma.

file_id_list = [
    'IuLDA9j3Ssac34DmLfECdQ',
    'T_NIjpy3ST23MR7Dv92DgQ',
    'JNoTCQ28QcG6YVaB6vJn-g',
    'J_-4mVrmQLKLnwklksH_yA',
]

Up to n=9, MB


file_id_list = [
    'VTLzByl7QvizwL8Q8-pXOg',
    '2aX98VL1QjOM9qp0-DywcA',
    'hxb7X9VkQ-OcHXtoFSfMoQ',
    '69pe4ucFQ2KI2_hjVzFQAw',
]
"""
"""
Fitting parameters, for reference.
(gamma, shift)

All combinations up to nmax = 5.
Only for first sample!

    (0.02107409, -0.02055823)

5 states, no BE distribution, 4 gammas.

    (0.02110444, -0.02056019),
    (0.01805621, -0.01317165),
    (0.02110444, -0.02056019),
    (0.01904502, -0.00823332),

5 states, BE distribution, 4 gammas.

    (0.02254297, -0.01887017),
    (0.01993866, -0.01153237),
    (0.02441793, -0.00987292),
    (0.02207004, -0.00703827),

5 states, BE distribution, 1 gamma.

    (0.02275155, -0.01887947),
    (0.02275155, -0.01166034),
    (0.02275155, -0.01008527),
    (0.02275155, -0.00699868),

Up to n=9, BE, 1 gamma.

    (0.02250705, -0.01889137),
    (0.02250705, -0.01166199),
    (0.02250705, -0.01009757),
    (0.02250705, -0.00700863),

Covariance matrix only.
[
    [
        6.31948701e-08,
        -3.53104014e-09,
        -2.28616172e-09,
        5.80545755e-09,
        6.67260015e-09,
    ],
    [
        -3.53104014e-09,
        5.52084698e-08,
        1.27740255e-10,
        -3.24382401e-10,
        -3.72834361e-10,
    ],
    [
        -2.28616172e-09,
        1.27740255e-10,
        5.52863133e-08,
        -2.10020447e-10,
        -2.41390528e-10,
    ],
    [
        5.80545755e-09,
        -3.24382401e-10,
        -2.10020447e-10,
        4.77113301e-08,
        6.12984834e-10,
    ],
    [
        6.67260015e-09,
        -3.72834361e-10,
        -2.41390528e-10,
        6.12984834e-10,
        4.64809801e-08,
    ],
]

MB, short, 5 states
[ 0.02188345 -0.01957042 -0.01245994 -0.01061118 -0.00754677]
[[ 7.13640980e-08 -3.74542579e-09 -2.10398922e-09  7.02974660e-09
  -7.09030358e-09]
 [-3.74542579e-09  6.29421065e-08  1.10424369e-10 -3.68944539e-10
   3.72122771e-10]
 [-2.10398922e-09  1.10424369e-10  6.36059898e-08 -2.07254229e-10
   2.09039598e-10]
 [ 7.02974660e-09 -3.68944539e-10 -2.07254229e-10  5.29438111e-08
  -6.98432950e-10]
 [-7.09030358e-09  3.72122771e-10  2.09039598e-10 -6.98432950e-10
   5.61843590e-08]]


MB, full, 5 states
[ 0.02161466 -0.01955469 -0.01245514 -0.01060767 -0.00752457]
[[ 3.76431422e-08 -2.03024730e-09 -1.14729617e-09  3.29953221e-09
  -3.89092776e-09]
 [-2.03024730e-09  3.44975027e-08  6.18783345e-11 -1.77957152e-10
   2.09853512e-10]
 [-1.14729617e-09  6.18783345e-11  3.48925513e-08 -1.00563886e-10
   1.18588573e-10]
 [ 3.29953221e-09 -1.77957152e-10 -1.00563886e-10  2.97786837e-08
  -3.41051271e-10]
 [-3.89092776e-09  2.09853512e-10  1.18588573e-10 -3.41051271e-10
   3.05714982e-08]]
"""


file_id_list = [
    'sykV12o6Ty-UgQ0gdf5NbQ',
    'eBFHb6IwSxCw-fP61faLDA',
    '9RExzuAZSJatRBuTiOl1fQ',
    'o_ETwXT0QjWPyLfsFtZf1Q',
]

for ii, file_id in enumerate(file_id_list):
    """
    data is an array of dimensions (N_states, N_energies, 2), where the last
    axis corresponds to the data (index 0) and the error (index 1).

    The PL data is obtained from PL_data = sum(data[:,:,0], axis=0).

    PL data is not normalized! Use PL_data /= amax(PL_data) to normalize.

    data_at_fit is computed at the same energies as measurements (E_vec_at_fit).

    extra_dict is a dictionary with the parameters used in the computation:
        'size_Lx'
        'size_Ly'
        'hwhm_x'
        'hwhm_y'
        'gamma'
        'shift'
        'states_vec'
        'E_vec'
    """
    E_vec, data, E_vec_at_fit, data_at_fit, extra_dict, _ = load_PL(
        '.',
        file_id,
    )

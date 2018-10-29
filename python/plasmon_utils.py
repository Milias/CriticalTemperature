from common import *

## Define how to pickle plasmon_elem_s objects

register_pickle_custom(plasmon_elem_s, 'id', 'wkwk', 'val')

class PlasmonPotcoefIterator:
  def filter(self, wkk):
    return wkk[1] < wkk[2] or wkk[0] < 0

  def __init__(self, w_vec, k_vec):
    self.w_vec, self.k_vec = w_vec, k_vec
    self.N_w, self.N_k = self.w_vec.size, self.k_vec.size
    self.__length_hint__ = self.N_w * self.N_k * (self.N_k + 1) // 2

    self.wkk_iter = itertools.filterfalse(
      self.filter,
      itertools.product(self.w_vec, self.k_vec, self.k_vec)
    )

  def __iter__(self):
    return self

  def __next__(self):
    return next(self.wkk_iter)

class PlasmonGreenMatrixIterator:
  def __init__(self, data, w_vec, k_vec):
    self.w_vec, self.k_vec = w_vec, k_vec
    self.N_w, self.N_k = self.w_vec.size, self.k_vec.size

    self.data = data.reshape((self.N_w, self.N_k * (self.N_k + 1) // 2, 2))

    r_w, r_k = range(self.N_w), range(self.N_k)

    self.ids_prod = itertools.product(r_w, r_k, repeat = 2)
    self.wkwk_prod = itertools.product(self.w_vec, self.k_vec, repeat = 2)

  def __iter__(self):
    return self

  def __next__(self):
    # i -> w0
    # k -> k0
    # j -> w1
    # l -> k1
    #
    # Will raise StopIteration when the iterator is exhausted.

    i, k, j, l = next(self.ids_prod)
    w0, k0, w1, k1 = next(self.wkwk_prod)

    if j > i:
      if l > k:
        return self.data[j - i, l - k, 0] + 1j * self.data[j - i, l - k, 1]
      else:
        return self.data[j - i, k - l, 0] + 1j * self.data[j - i, k - l, 1]
    else:
      if l > k:
        return self.data[i - j, l - k, 0] - 1j * self.data[i - j, l - k, 1]
      else:
        return self.data[i - j, k - l, 0] - 1j * self.data[i - j, k - l, 1]


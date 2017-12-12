import ctypes

### Global

class genericFunctor:
  def __init__(self, libc, func_name, argtypes, restype):
    self.func = libc.__getattr__(func_name)
    self.func.argtypes = argtypes
    self.func.restype = restype

  def __call__(self, *args):
    return self.func(*args)

integrals_so = ctypes.CDLL('bin/libintegrals.so')

### common.h

class logExpClass(genericFunctor):
  def __call__(self, x, xmax = 50):
    try:
      return self.func(x, xmax)
    except Exception as e:
      print(e)
      return

logExp = logExpClass(integrals_so, "logExp", [ ctypes.c_double, ctypes.c_double ], ctypes.c_double)

### integrals.h

I1 = genericFunctor(integrals_so, "I1", [ ctypes.c_double ], ctypes.c_double)
I1dmu = genericFunctor(integrals_so, "I1dmu", [ ctypes.c_double ], ctypes.c_double)

I2_real = genericFunctor(integrals_so, "integralI2Real", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)
I2_imag = genericFunctor(integrals_so, "integralI2Imag", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

def I2(w, E, mu, beta):
  return complex(I2_real(w, E, mu, beta), I2_imag(w, E, mu, beta))

polePos = genericFunctor(integrals_so, "polePos", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)


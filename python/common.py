import time
import copyreg
import itertools

from numpy import *
import matplotlib.pyplot as plt

from semiconductor import *
from job_api import JobAPI

initializeMPFR_GSL()

## Define how to pickle system_data objects

def pickle_system_data(sys):
  return system_data, (sys.dl_m_e, sys.dl_m_h, sys.eps_r, sys.T)

copyreg.pickle(system_data, pickle_system_data)


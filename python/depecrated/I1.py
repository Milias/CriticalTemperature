from common import *

def I1(w, E, mu):
  return - 0.5 * sm.sqrt(E - 4 * mu - 2 * w)

def I1dmu(w, E, mu):
  return 2 / sm.sqrt(E - 4 * mu - 2 * w)


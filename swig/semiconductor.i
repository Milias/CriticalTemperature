#define SWIGPYTHON_BUILTIN

%module semiconductor

%include "stdint.i"
%include "std_complex.i"
%include "std_vector.i"

%template(DoubleVector) std::vector<double>;

%{
#include "common.h"
#include "utils.h"
#include "wavefunction.h"
#include "analytic.h"
#include "fluctuations.h"
#include "analytic_2d.h"
%}

%include "common.h"
%include "utils.h"
%include "wavefunction.h"
%include "analytic.h"
%include "fluctuations.h"
%include "analytic_2d.h"


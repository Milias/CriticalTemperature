#define SWIGPYTHON_BUILTIN

%module semiconductor

%include "stdint.i"
%include "std_complex.i"
%include "std_vector.i"

%template(DoubleVector) std::vector<double>;

%{
#include "common.h"
#include "utils.h"
#include "analytic.h"
#include "fluctuations.h"
%}

%include "common.h"
%include "utils.h"
%include "analytic.h"
%include "fluctuations.h"


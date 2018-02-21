%module semiconductor

%include "stdint.i"
%include "std_complex.i"
%include "std_vector.i"

namespace std {
   %template(DoubleVector) vector<double>;
}

%{
#include "common.h"
#include "utils.h"
#include "analytic.h"
%}

%include "common.h"
%include "utils.h"
%include "analytic.h"


%module integrals

%include "stdint.i"
%include "std_vector.i"
namespace std {
   %template(DoubleVector) vector<double>;
}

%{
#include "common.h"
#include "integrals.h"
#include "expansion.h"
%}

%include "common.h"
%include "integrals.h"
%include "expansion.h"


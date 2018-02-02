%module integrals
%{
#include "common.h"
#include "integrals.h"
%}

%include "std_vector.i"
namespace std {
   %template(DoubleVector) vector<double>;
}

%include "common.h"
%include "integrals.h"


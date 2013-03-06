%module mwumkl

%{
#define SWIG_FILE_WITH_INIT
#include "../mwu_main.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%include "../mwu_main.h"


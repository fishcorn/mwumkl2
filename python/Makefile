# SWIG
INTERFACES = mwu_main.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = mwumkl.py #$(INTERFACES:.i=.py      )

# Default target: build the extension
# Add $(PROXIES) as a guard against braindead swig compilation
# 'make swig' should be done explicitly before 'make' or 'make test'
.PHONY : all
all: $(PROXIES) _mwumkl.so 

_mwumkl.so: $(WRAPPERS) ../mwu_dynamic.cpp ../kernel.hpp ../mwu_main.h
	./setup.py build_ext -i

# Test target: run the tests
.PHONY : test
test: all 
	python testTestMKL.py

# Rule: %.i -> %_wrap.cxx
%_wrap.cxx: %.i ../%.h numpy.i
	swig -c++ -python $<
%_wrap.cxx: %.i ../%1.h ../%2.h numpy.i
	swig -c++ -python $<

# Clean target
.PHONY : clean
clean:
	$(RM) *.pyc
	$(RM) -r build
	$(RM) *.so

# USE CAUTION FOR TARGETS AFTER THIS POINT
# clean everything
.PHONY : veryclean
veryclean: clean swigclean

# swig target: run swig to build the wrappers
.PHONY : swig
swig: $(WRAPPERS)

# Clean swig generated files
.PHONY : swigclean
swigclean:
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)

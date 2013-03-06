# SWIG
INTERFACES = mwu_main.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = mwumkl.py $(INTERFACES:.i=.py      )

# Default target: build the tests
.PHONY : all
all: $(WRAPPERS) ../mwu_dynamic.cpp ../kernel.hpp ../mwu_main.h
	./setup.py build_ext -i

# Test target: run the tests
.PHONY : test
test: all
	# python testVector.py
	# python testMatrix.py
	# python testTensor.py
	# python testArray.py
	# python testFarray.py
	# python testFortran.py

# Rule: %.i -> %_wrap.cxx
%_wrap.cxx: %.i ../%.h numpy.i
	swig -c++ -python $<
%_wrap.cxx: %.i ../%1.h ../%2.h numpy.i
	swig -c++ -python $<

# Clean target
.PHONY : clean
clean:
	$(RM) -r build
	$(RM) *.so
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)

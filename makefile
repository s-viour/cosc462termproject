all:
	mkdir build && cd build && cmake -G"Unix Makefiles" -DCMAKE_CXX_COMPILER=mpicxx .. && make

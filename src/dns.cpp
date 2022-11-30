#include <iostream>
#include <stdexcept>
#include <cmath>
#include <mpi.h>
#include <matrix.h>


struct point {
	int i;
	int j;
	int k;
};

/// @brief get the processor mapped to location in the (n*n*n) cube (i, j, k)
/// @param i row
/// @param j column
/// @param k pillar
/// @param n dimension of cube
/// @return processor rank mapped to location (i, j, k)
int map_processor_to_1d(int i, int j, int k, int n);

/// @brief get the index (i, j, k) of the processor `p` 
/// @param p processor number
/// @return `point` of processor
point map_processor_to_3d(int p, int n);

void print_matrix(const matrix&);


int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	// argument checking & parsing
	if (argc != 2) {
		std::cerr << "usage: " << argv[0] << " n\n";
		return EXIT_FAILURE;
	}

	int n;
	try {
		n = std::stoi(argv[1]);
	} catch (const std::invalid_argument& e) {
		std::cerr << "error: n must be an integer\n";
		return EXIT_FAILURE;
	} catch (const std::out_of_range& e) {
		std::cerr << "error: n out of range\n";
		return EXIT_FAILURE;
	}

	// get nranks and myrank
	int nranks, myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	// size of each submatrix will be n / p
	//   where p = sqrt(P)
	int p = std::cbrt(nranks);
	int submatrix_size = n / p;

	matrix my_mtx1(submatrix_size);
	matrix my_mtx2(submatrix_size);
	auto my_point = map_processor_to_3d(myrank, p);
	
	// partition matrices exactly like cannon's algorithm at first
	// do this among the bottom layer (k = 0)
	if (myrank == 0) {
		auto mtx1 = matrix(n);
		auto mtx2 = matrix(n);

		my_mtx1 = mtx1.submatrix(submatrix_size, 0, 0);
		my_mtx2 = mtx2.submatrix(submatrix_size, 0, 0);

		for (int i = 0; i < p; ++i) {
			for (int j = 0; j < p; ++j) {
				// skip first iteration
				if (i == 0 && j == 0) {
					continue;
				}

				// set k = 0;
				int k = 0;
				int proc = map_processor_to_1d(i, j, k, p);

				auto submtx1 = mtx1.submatrix(submatrix_size, i, j);
				auto submtx2 = mtx2.submatrix(submatrix_size, i, j);
				MPI_Send(submtx1.data(), submtx1.total_elems(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
				MPI_Send(submtx2.data(), submtx2.total_elems(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
			}
		}
	} else if (my_point.k == 0) {
		//std::clog << "processor " << myrank << " getting submatrix\n";
		MPI_Recv(my_mtx1.data(), my_mtx1.total_elems(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(my_mtx2.data(), my_mtx2.total_elems(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// at this point, all the submatrices are partitioned along layer 0
	// now, we need to copy the jth column of A to the (k = j)th layer
	if (my_point.k == 0) {
		if (my_point.j != 0) {
			int proc = map_processor_to_1d(my_point.i, my_point.j, my_point.j, p);
			MPI_Send(my_mtx1.data(), my_mtx1.total_elems(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
		}
	} else {
		// receive from bottom layer
		if (my_point.i != 0) {
			int src = map_processor_to_1d(my_point.i, my_point.j, 0, p);
			MPI_Recv(my_mtx1.data(), my_mtx1.total_elems(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	// now we need to copy the ith column of B to the (k = ith) layer
	
	if (my_point.k == 0) {
		if (my_point.i != 0) {
			int proc = map_processor_to_1d(my_point.i, my_point.j, my_point.i, p);
			MPI_Send(my_mtx2.data(), my_mtx2.total_elems(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
		}
	} else {
		// receive from bottom layer
		if (my_point.j != 0) {
			int src = map_processor_to_1d(my_point.i, my_point.j, 0, p);
			MPI_Recv(my_mtx2.data(), my_mtx2.total_elems(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	// at this point, each processor should multiply its two matrices
	auto my_result = matrix_multiply(my_mtx1, my_mtx2);
	auto tmp_result = matrix(my_result);

	// collapse the tower
	for (int k = p - 1; k > 0; --k) {
		if (my_point.k == k) {
			// processor on upper level sends down one level
			int dest = map_processor_to_1d(my_point.i, my_point.j, my_point.k - 1, p);
			MPI_Send(my_result.data(), my_result.total_elems(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
		} else if (my_point.k == k - 1) {
			// processor on lower level receives and adds to its own result
			int src = map_processor_to_1d(my_point.i, my_point.j, my_point.k + 1, p);
			MPI_Recv(tmp_result.data(), tmp_result.total_elems(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			my_result = matrix_add(my_result, tmp_result);
		}
	}

	// once this loop is complete, each processor on level k=0
	// has their part of the full matrix, so now we just need to recombine
	if (myrank != 0 && my_point.k == 0) {
		MPI_Send(my_result.data(), my_result.total_elems(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	} else if (myrank == 0) {
		matrix final_result(n);
		final_result.set_submatrix(my_result, 0, 0);
		for (int i = 0; i < p; ++i) {
			for (int j = 0; j < p; ++j) {
				// skip first iteration since we've already handled processor 0
				if (i == 0 && j == 0) {
					continue;
				}

				int src = map_processor_to_1d(i, j, 0, p);
				MPI_Recv(my_result.data(), my_result.total_elems(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				final_result.set_submatrix(my_result, i, j);
			}
		}

		print_matrix(final_result);
	}


	MPI_Finalize();
	return EXIT_SUCCESS;
}

// implementation for these two functions sourced from 
// https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array
int map_processor_to_1d(int i, int j, int k, int n) {
	return (k * n * n) + (j * n) + i;
}

point map_processor_to_3d(int p, int n) {
	point pp;
	pp.k = p / (n * n);
	p -= (pp.k * n * n);
	pp.j = p % n;
	pp.i = p / n;

	return pp;
}

void print_matrix(const matrix& m) {
	for (int i = 0; i < m.size(); ++i) {
		for (int j = 0; j < m.size(); ++j) {
			std::cout << m.at(i, j) << ' ';
		}
		std::cout << '\n';
	}
}
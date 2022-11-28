#include <iostream>
#include <stdexcept>
#include <cmath>
#include <mpi.h>
#include <matrix.h>


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
  int p = std::sqrt(nranks);
  int submatrix_size = n / p;

  matrix my_mtx1(submatrix_size);
  matrix my_mtx2(submatrix_size);

  // processor 0 needs to create two big matrices and partition them
  if (myrank == 0) {
    auto mtx1 = matrix(n);
    auto mtx2 = matrix(n);

    // submatrix 0, 0 is ours
    my_mtx1 = mtx1.submatrix(submatrix_size, 0, 0);
    my_mtx2 = mtx2.submatrix(submatrix_size, 0, 0);

    // send the rest of the submatrices to higher ranked processors
    int proc = 0;
    int i = 0;
    int j = 0;
    for (int x = 0; x < nranks; ++x) {
      // skip first iteration since we've already handled that
      if (proc == 0) {
        proc += 1;
        // increment j since the next submatrix is (0, 1)
        j += 1;
        continue;
      }
      // divide up submatrices and send them
      auto submtx1 = mtx1.submatrix(submatrix_size, i, j);
      auto submtx2 = mtx2.submatrix(submatrix_size, i, j);
      MPI_Send(submtx1.data(), submtx1.total_elems(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      MPI_Send(submtx2.data(), submtx2.total_elems(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);

      if (j == p - 1) {
        j = 0;
        i += 1;
      } else {
        j += 1;
      }
      proc += 1;
    }
  } else {
    // if we AREN'T processor 0, receive everything from previous processors
    MPI_Recv(my_mtx1.data(), my_mtx1.total_elems(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(my_mtx2.data(), my_mtx2.total_elems(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  std::cout << "processor " << myrank << " has matrix\n";
  print_matrix(my_mtx1);

  MPI_Finalize();
  return EXIT_SUCCESS;
}


void print_matrix(const matrix& m) {
  for (int i = 0; i < m.size(); ++i) {
    for (int j = 0; j < m.size(); ++j) {
      std::cout << m.at(i, j) << ' ';
    }
    std::cout << '\n';
  }
}
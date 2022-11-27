#include <iostream>
#include <stdexcept>
#include <mpi.h>
#include <matrix.h>


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

  // create matrices
  auto mtx1 = matrix(n);
  auto mtx2 = matrix(n);

  // print matrix out
  auto submtx = mtx1.submatrix(n / 4, 0, 0);
  for (int i = 0; i < submtx.size(); ++i) {
    for (int j = 0; j < submtx.size(); ++j) {
      std::cout << submtx.at(i, j) << ' ';
    }
    std::cout << '\n';
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

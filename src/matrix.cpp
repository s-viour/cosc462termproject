#include <stdexcept>
#include <matrix.h>


const int matrix::size() {
  return this->_n;
}

const double& matrix::at(int r, int c) {
  return this->_data[c + this->_n * r];
}

double& matrix::get(int r, int c) {
  return this->_data[c + this->_n * r];
}

matrix matrix::submatrix(int n, int r, int c) {
  int partitions = this->size() / n;
  if (r >= partitions || c >= partitions) {
    throw std::out_of_range("row or column out of range");
  }

  matrix mtx(n);
  int mtx_i = 0;
  int mtx_j = 0;
  for (int i = r * n; i < (r * n) + n; ++i) {
    for (int j = c * n; j < (c * n) + n; ++j) {
      mtx.get(mtx_i, mtx_j) = this->at(i, j);
      mtx_j++;
    }
    mtx_i++;
    mtx_j = 0;
  }

  return mtx;
}

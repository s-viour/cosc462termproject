#include <stdexcept>
#include <matrix.h>


double* matrix::data() {
  return this->_data.data();
}

const int matrix::size() const {
  return this->_n;
}

const int matrix::total_elems() const {
  return this->size() * this->size();
}

const double& matrix::at(int r, int c) const {
  return this->_data[c + this->_n * r];
}

double& matrix::get(int r, int c) {
  return this->_data[c + this->_n * r];
}

void matrix::set_submatrix(const matrix& submatrix, int r, int c) {
  int partitions = this->size() / submatrix.size();
  if (r >= partitions || c >= partitions) {
    throw std::out_of_range("row or column out of range");
  }

  // mapping a submatrix onto a matrix
  // lower bound of row: (r * this->size()) / partitions
  // upper bound of row: lower_bound + (this->size() / partitions)
  int rl = (r * this->size()) / partitions;
  int ru = rl + (this->size() / partitions);
  int cl = (c * this->size()) / partitions;
  int cu = cl + (this->size() / partitions);
  int si = 0, sj = 0;
  for (int i = rl; i < ru; ++i) {
    for (int j = cl; j < cu; ++j) {
      this->get(i, j) = submatrix.at(si, sj);
      sj += 1;
    }
    si += 1;
    sj = 0;
  }
}

matrix matrix::submatrix(int n, int r, int c) const {
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

matrix matrix_add(const matrix& a, const matrix& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("matrices A and B of different sizes");
  }

  // copy a into c
  matrix c(a);

  // add
  for (int i = 0; i < a.size(); ++i) {
    for (int j = 0; j < b.size(); ++j) {
      c.get(i, j) += b.at(i, j);
    }
  }

  return c;
}

matrix matrix_multiply(const matrix& a, const matrix& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("matrices A and B of different sizes");
  }

  // create our output matrix
  matrix c(a.size(), 0);
  for (int i = 0; i < a.size(); ++i) {
    for (int j = 0; j < a.size(); ++j) {
      for (int k = 0; k < a.size(); ++k) {
        c.get(i, j) += a.at(i, k) * b.at(k, j);
      }
    }
  }

  return c;
}
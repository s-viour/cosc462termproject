#include <vector>


/// @brief represents a square matrix of size `n*n` in a contiguous memory block
///
class matrix {
private:
  int _n;
  std::vector<double> _data;

public:
  /// @brief construct a blank matrix
  matrix() {}

  /// @brief construct an `n*n` matrix of ones
  /// @param  n size of matrix
  matrix(int n, double val=1) : _n(n), _data(std::vector<double>(n * n, val)) {};

  /// @brief construct an `n*n` matrix from a pointer to a list of elements
  /// @param  new_data data to construct the matrix from
  /// @param  n size of the matrix being constructed
  matrix(double* new_data, int n) : _n(n), _data(std::vector<double>(new_data, new_data + n * n)) {};

  /// @brief get the size of the matrix
  /// @return size
  const int size() const;

  /// @brief get the total elements in the matrix (equivalent to size() * size())
  /// @return total number of elements in the matrix
  const int total_elems() const;

  /// @brief get the data inside the matrix
  /// @return pointer to the matrix data
  double* data();

  /// @brief return a constant reference to an element
  /// @param  r row of element
  /// @param  c column of element
  /// @return element at position `(c, r)`
  const double& at(int r, int c) const;

  /// @brief return a non-const reference to an element
  /// @param r row of element
  /// @param c column of element
  /// @return element at position `(c, r)`
  double& get(int r, int c);

  /// @brief get the submatrix of size `n` at row `r` and column `c`
  /// @param  n size of submatrix
  /// @param  r row of submatrix
  /// @param  c column of submatrix
  /// @return submatrix at position `(c, r)` of size `n`
  matrix submatrix(int n, int r, int c) const;
};


matrix matrix_multiply(const matrix&, const matrix&);

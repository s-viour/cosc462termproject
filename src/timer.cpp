#include <timer.h>


void timer::start_compute() {
  this->_tcomp = MPI_Wtime();
}

void timer::end_compute() {
  double t2 = MPI_Wtime();
  this->_compute_time += (t2 - this->_tcomp);
}

void timer::start_communication() {
  this->_tcomm = MPI_Wtime();
}

void timer::end_communication() {
  double t2 = MPI_Wtime();
  this->_communication_time += (t2 - this->_tcomm);
}

double timer::total_compute_time() {
  return this->_compute_time;
}

double timer::total_communication_time() {
  return this->_communication_time;
}

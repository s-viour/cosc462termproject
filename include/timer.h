#include <mpi.h>


class timer {
private:
  double _compute_time;
  double _communication_time;
  double _tcomp;
  double _tcomm;

public:
  timer() : _compute_time(0), _communication_time(0) {}

  void start_compute();
  void start_communication();
  void end_compute();
  void end_communication();
  double total_compute_time();
  double total_communication_time();  
};

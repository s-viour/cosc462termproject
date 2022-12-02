#include <mpi.h>


class timer {
private:
  double _compute_time;
  double _communication_time;
  double _tcomp;
  double _tcomm;

public:
  timer() : _compute_time(0), _communication_time(0) {}

  /// @brief begin counting computation time
  void start_compute();

  /// @brief begin counting communication time
  void start_communication();

  /// @brief stop counting computation time (for now)
  void end_compute();

  /// @brief stop counting communication time (for now)
  void end_communication();

  /// @brief get the total compute time counted so far
  /// @return compute time in seconds
  double total_compute_time();

  /// @brief get the total communication time counted so far
  /// @return communication time in seconds
  double total_communication_time();  
};

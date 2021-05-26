#ifndef SIPML_SRC_TM_ESTIMATOR_HH_
#define SIPML_SRC_TM_ESTIMATOR_HH_
#include <cstdint>
#include "sim_config.hh"
#include "session.hh"

class TMEstimatorBase {
 public:
  Matrix2D< double > tm_est;
  const int num_gpus;
  Session *sessions;
  int num_sessions;
 protected:
  const std::string log_dir;
 private:
  std::ofstream log_file;
 public:
  TMEstimatorBase( const int num_gpus, const std::string log_dir ) : tm_est( num_gpus, num_gpus ),
                                                                     num_gpus( num_gpus ),
                                                                     sessions( nullptr ),
                                                                     num_sessions( 0 ),
                                                                     log_dir( log_dir ), 
                                                                     log_file( ) {
    log_file = std::ofstream( log_dir + "tm_estimator.log", std::ofstream::out );
    tm_est.fill_zeros( );
  }

  virtual ExitStatus update_tm_est( ) = 0;

  virtual ~TMEstimatorBase( ) = default;

  TMEstimatorBase(const TMEstimatorBase&) = delete;

  TMEstimatorBase &operator=(const TMEstimatorBase & ) = delete;

 public:
  ExitStatus bind_to_sessions( Session *sess, int n_sess );

 public:
  void log( );
};

class SingleShotEsimator : public TMEstimatorBase {
 public:
  SingleShotEsimator( const int num_gpus, const std::string &log_dir );

  ExitStatus update_tm_est( ) override;

  virtual ~SingleShotEsimator( ) = default;
};

class TransportEstimator : public TMEstimatorBase {
  Transport **transports;
  int num_transports;
 public:
  TransportEstimator( const int num_gpus, const std::string &log_dir ) : TMEstimatorBase(
      num_gpus, log_dir ), transports( nullptr ), num_transports( -1 ) {
    transports = new Transport *[num_gpus];
  }

  ExitStatus update_tm_est( ) override;

  ExitStatus set_eff_num_transports( int n );

  ExitStatus bind_to_transports( GPU *gpus, int num_gpus );

  virtual ~TransportEstimator( ) = default;

  TransportEstimator(const TransportEstimator&) = delete;

  TransportEstimator &operator=(const TransportEstimator&) = delete;
};

#endif //SIPML_SRC_TM_ESTIMATOR_HH_

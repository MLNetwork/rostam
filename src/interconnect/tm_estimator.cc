#include "tm_estimator.hh"

ExitStatus TMEstimatorBase::bind_to_sessions( Session *sess, int n_sess ) {
  sessions = sess;
  num_sessions = n_sess;
  NetOp *net_op;
  for ( int i = 0; i < num_sessions; i ++ ) {
    for ( auto e : sessions[ i ].graph.adj ) {
      if ( e.first->device->type == DeviceType::INTERCONNECT ) {
        net_op = static_cast< NetOp * >( e.first );
        tm_est.add_elem_by( net_op->src_device->dev_id, net_op->dst_device->dev_id, net_op->transfer_bytes );
      }
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus SingleShotEsimator::update_tm_est( ) {
  /* nothing to be done :D */
  return ExitStatus::SUCCESS;
}

SingleShotEsimator::SingleShotEsimator( const int num_gpus, const std::string &log_dir ) : TMEstimatorBase( num_gpus,
                                                                                                            log_dir ) { }

void TMEstimatorBase::log( ) {
  log_file << tm_est;
}

ExitStatus TransportEstimator::update_tm_est( ) {
  Matrix2D< double > temp_tm( num_transports, num_transports );
  for ( int tp_no = 0; tp_no < num_transports; tp_no ++ ) {
    transports[ tp_no ]->get_tm_estimate( temp_tm ); //todo: make this more efficient by seperating into sent & recv bytes
    tm_est.add_by( temp_tm );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus TransportEstimator::bind_to_transports( GPU *gpus, int num_gpus ) {
  num_transports = num_gpus;
  for ( int gpu_no = 0; gpu_no < num_gpus; gpu_no ++ ) {
    transports[ gpu_no ] = gpus[ gpu_no ].tp;
  }
  update_tm_est( );
  return ExitStatus::SUCCESS;
}

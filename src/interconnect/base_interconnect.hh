#ifndef TEST_INTERCONNECT_H
#define TEST_INTERCONNECT_H
#include <deque>
#include <assert.h>
#include "packet.hh"
#include "gpu.hh"
#include "tm_estimator.hh"
#include "config.h"

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif //HAVE_GUROBI

class BaseInterconnect : public Device {
 public:
  GPU *gpus;
 public:
  uint16_t num_gpus;
 private:
  std::deque< Packet * > **to_send_buff;
  std::deque< Packet * > *to_recv_buff;
//  Matrix2D< double > step_bytes_budget;
  const double ingress_link_speed;
  const double egress_link_speed;
 protected:
  Matrix2D< double > episode_bw;
  std::map< uint16_t, std::map< uint16_t, double > > sparse_episode_bw;
 public:
  TMEstimatorBase *tm_estimator;
  SimConfig cnfg;
 private:
  uint64_t total_bytes_transferred;
 protected:
  const std::string log_dir;
 public:
  BaseInterconnect( uint16_t dev_id,
                    GPU *gpus,
                    uint16_t num_gpus,
                    double ingress_link_speed,
                    double egress_link_speed,
                    TMEstimatorBase *tm_estimator,
                    const SimConfig &cnfg,
                    const std::string log_dir ) : Device( dev_id, DeviceType::INTERCONNECT ),
                                                  gpus( gpus ),
                                                  num_gpus( num_gpus ),
                                                  to_send_buff( ),
                                                  to_recv_buff( ),
//                                                  step_bytes_budget( num_gpus, num_gpus ),
                                                  ingress_link_speed( ingress_link_speed ),
                                                  egress_link_speed( egress_link_speed ),
                                                  episode_bw( num_gpus, num_gpus ),
                                                  tm_estimator( tm_estimator ),
                                                  cnfg( cnfg ),
                                                  total_bytes_transferred( 0 ),
                                                  log_dir( log_dir ) {
    to_recv_buff = new std::deque< Packet * >[num_gpus];
    to_send_buff = new std::deque< Packet * > *[num_gpus];
    for ( int gpu_no = 0; gpu_no < num_gpus; gpu_no ++ ) {
      to_send_buff[ gpu_no ] = new std::deque< Packet * >[num_gpus];
    }
  }

  ExitStatus proceed( );

  ExitStatus report_episode_bw( );

  /* offline_bw_est( ) provides an estimate of bw for all GPU pairs,
   * helpful for e.g., faster placement strategy decisions;
   * should NOT be used for any run-time purposes. */
  virtual ExitStatus offline_bw_est( std::unordered_map< Device *,
                                                         std::unordered_map< Device *, double>> &estimate ) = 0;

  virtual ExitStatus is_routing_feasible( Packet* pkt, bool &is_bw_avail ) = 0;

  BaseInterconnect( const BaseInterconnect & ) = delete;

  BaseInterconnect( BaseInterconnect && ) = delete;

  BaseInterconnect &operator=( const BaseInterconnect & ) = delete;

  BaseInterconnect &operator=( BaseInterconnect && ) = delete;

  virtual ~BaseInterconnect( ) = default;

 protected:
  ExitStatus normalize_tm( Matrix2D< double > &normal_tm ) const;

 private:
  ExitStatus proceed_ingress( );

  ExitStatus proceed_egress( );

  ExitStatus proceed_routing( );

  ExitStatus allocate_step_bw( );

  virtual ExitStatus allocate_episode_bw( ) = 0;

  virtual ExitStatus reset_routing_step_counters( );

  ExitStatus progress_log( );
};

#endif //TEST_INTERCONNECT_H

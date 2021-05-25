#include <iostream>
#include <stdexcept>
#include "base_interconnect.hh"
#define MAX_NUM_GPUS 4096

ExitStatus BaseInterconnect::proceed_ingress( ) {
  bool ingress_rate_cond;
  uint64_t ingress_bytes_budget[MAX_NUM_GPUS];
  uint64_t max_step_bytes = ingress_link_speed * cnfg.step_size_sec; //ToDo: double check
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    ingress_bytes_budget[i] = max_step_bytes;
  }
  Packet *pkt;
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    ingress_rate_cond = true;
    while ( ingress_rate_cond ) {
      if ( gpus[ i ].fetch_tx( pkt ) == ExitStatus::SUCCESS ) {
        assert( pkt->tx_time == curr_step );
        assert( pkt->src->dev_id == i );
        assert( pkt->dst->dev_id != i );
        to_send_buff[ pkt->src->dev_id ][ pkt->dst->dev_id ].push_back( pkt );
        ingress_bytes_budget[ pkt->src->dev_id ] -= pkt->num_bytes;
        ingress_rate_cond = ( ingress_bytes_budget[ pkt->src->dev_id ] >= pkt->num_bytes );
      } else break;
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::proceed_egress( ) {
  bool egress_rate_cond;
  uint64_t egress_bytes_budget[MAX_NUM_GPUS];
  uint64_t max_step_bytes = egress_link_speed * cnfg.step_size_sec; //ToDo: double check
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    egress_bytes_budget[i] = max_step_bytes;
  }
  Packet *pkt;
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    egress_rate_cond = true;
    while ( ( ! to_recv_buff[ i ].empty( ) ) && egress_rate_cond ) {
      pkt = to_recv_buff[ i ].front( );
      assert( pkt->rx_time == curr_step );
      egress_bytes_budget[ pkt->dst->dev_id ] -= pkt->num_bytes;
      egress_rate_cond = ( egress_bytes_budget[ pkt->dst->dev_id ] >= pkt->num_bytes );
      to_recv_buff[ i ].pop_front( );
      gpus[ i ].fill_rx( pkt, 1 ).ok( ); //fill_rx removes the packet from the memory; so call it the at the end.
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::reset_routing_step_counters( ){
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::proceed_routing( ) {
  reset_routing_step_counters( ).ok( ); // can be used for rate limiting, etc.
  bool is_feasible;
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      while ( ! to_send_buff[ i ][ j ].empty( )) {
        Packet *cand_pkt = to_send_buff[ i ][ j ].front( );
        assert( cand_pkt->src->dev_id == i );
        assert( cand_pkt->dst->dev_id == j );
        is_routing_feasible( cand_pkt, is_feasible );
        bool is_lat_met = ( cand_pkt->tx_time + cnfg.interconnect_latency ) <= Device::curr_step;
        if ( is_feasible && is_lat_met ) {
//          step_bytes_budget.sub_elem_by( i, j, cand_pkt->num_bytes );
          total_bytes_transferred += cand_pkt->num_bytes;
          cand_pkt->rx_time = curr_step;
          to_recv_buff[ j ].push_back( cand_pkt );

          to_send_buff[ i ][ j ].pop_front( );
        } else {
          break;
        }
      }
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::allocate_step_bw( ) {
//  step_bytes_budget.copy_from( episode_bw );
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::proceed( ) {
  if ( curr_step % ( cnfg.dec_interval + cnfg.interconnect_reconf_delay )== 0 ) {
    tm_estimator->update_tm_est( ).ok( );
    allocate_episode_bw( ).ok( );
  }

//  allocate_step_bw( ).ok( );
  proceed_ingress( ).ok( );
  proceed_routing( ).ok( );
  proceed_egress( ).ok( );
  if ( Device::curr_step % 1000 == 0 )
    progress_log( ).ok( );
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::report_episode_bw( ) {
  std::cout << episode_bw;
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::normalize_tm( Matrix2D< double > &normal_tm ) const {
  normal_tm.copy_from( tm_estimator->tm_est );
  normal_tm.normalize_by_max( );
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::progress_log( ) {
  std::cout << "[ interconnect ]" << " "
            << "curr_step=" << Device::curr_step << " "
            << "total_bytes_transferred=" << total_bytes_transferred << " "
            << std::endl;
  return ExitStatus::SUCCESS;
}

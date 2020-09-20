#include <iostream>
#include <stdexcept>
#include "base_interconnect.hh"

ExitStatus BaseInterconnect::read_tx_buffers( ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    while ( true ) {
      Packet *pkt;
      if ( gpus[ i ].fetch_tx( pkt ) == ExitStatus::SUCCESS ) {
        assert( pkt->tx_time == curr_step );
        assert( pkt->src->dev_id == i );
        assert( pkt->dst->dev_id != i );
        to_send_buff[ pkt->src->dev_id ][ pkt->dst->dev_id ].push_back( pkt );
      } else break;
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::write_rx_buffers( ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    while ( ! to_recv_buff[ i ].empty( )) {
      Packet *pkt = to_recv_buff[ i ].front( );
      assert( pkt->rx_time == curr_step );
      gpus[ i ].fill_rx( pkt, 1 ).ok( );
      to_recv_buff[ i ].pop_front( );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::route( ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      while ( ! to_send_buff[ i ][ j ].empty( )) {
        Packet *cand_pkt = to_send_buff[ i ][ j ].front( );
        assert( cand_pkt->src->dev_id == i );
        assert( cand_pkt->dst->dev_id == j );
        bool is_bw_avail = ( step_bytes_budget.get_elem( i, j ) >= cand_pkt->num_bytes );
        bool is_lat_met = ( cand_pkt->tx_time + cnfg.interconnect_latency ) <= Device::curr_step;
        if ( is_bw_avail && is_lat_met ) {
          step_bytes_budget.sub_elem_by( i, j, cand_pkt->num_bytes );
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
  step_bytes_budget.copy_from( episode_bw );
  return ExitStatus::SUCCESS;
}

ExitStatus BaseInterconnect::proceed( ) {
  if ( curr_step % cnfg.dec_interval == 0 ) {
    tm_estimator->update_tm_est( ).ok( );
    allocate_episode_bw( ).ok( );
  }
  allocate_step_bw( ).ok( );
  read_tx_buffers( ).ok( );
  route( ).ok( );
  write_rx_buffers( ).ok( );
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

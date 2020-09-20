#include "op.hh"

void CompOp::copy_scale_to( CompOp *new_op, double batch_scale ) {
  new_op->name = std::move( name );
  new_op->type = type;
  new_op->device = device;
  new_op->session_id = session_id;
  new_op->comp_time_map = comp_time_map;
  new_op->output_bytes_map = output_bytes_map;
  new_op->comp_time = Step( double( comp_time ) * batch_scale );
  new_op->output_bytes = uint32_t( double( output_bytes ) * batch_scale );
}

ExitStatus CompOp::get_mem_size( uint64_t &size ) const {
  size = output_bytes;
  return ExitStatus::SUCCESS;
}

ExitStatus CompOp::get_batch_size( uint16_t &bs ) {
  bs = batch_size;
  return ExitStatus::SUCCESS;
}

ExitStatus CompOp::set_batch_size( uint16_t bs ) {
  if ( comp_time_map.empty( ))
    return ExitStatus::FAILURE;
  batch_size = bs;
  if ( comp_time_map.count( bs ) == 1 ) {
    comp_time = comp_time_map.at( bs );
    output_bytes = output_bytes_map.at( bs );
    return ExitStatus::SUCCESS;
  } else {
    comp_time = 0;
    output_bytes = 0;
    uint16_t first_peer_bs = 0;
    int first_peer_diff = std::numeric_limits< int >::max( );
    for ( auto peer : comp_time_map ) {
      int diff = int( bs ) - int( peer.first );
      diff = ( diff > 0 ? diff : - diff );
      if ( diff <= first_peer_diff ) {
        first_peer_diff = diff;
        first_peer_bs = peer.first;
      }
    }
    uint16_t second_peer_bs = 0;
    int second_peer_diff = std::numeric_limits< int >::max( );
    for ( auto peer : comp_time_map ) {
      int diff = int( bs ) - int( peer.first );
      diff = ( diff > 0 ? diff : - diff );
      if ( diff <= second_peer_diff && peer.first != first_peer_bs ) {
        second_peer_diff = diff;
        second_peer_bs = peer.first;
      }
    }
    assert( first_peer_bs != second_peer_bs );
    auto comp_time_1 = comp_time_map.at( first_peer_bs );
    auto output_bytes_1 = output_bytes_map.at( first_peer_bs );
    auto comp_time_2 = comp_time_map.at( second_peer_bs );
    auto output_bytes_2 = output_bytes_map.at( second_peer_bs );
    double time_slope =
        ( double( comp_time_2 ) - double( comp_time_1 )) / ( double( second_peer_bs ) - double( first_peer_bs ));
    double bytes_slope =
        ( double( output_bytes_2 ) - double( output_bytes_1 )) / ( double( second_peer_bs ) - double( first_peer_bs ));
    comp_time = comp_time_1 + time_slope * double( bs - first_peer_bs );
    output_bytes = output_bytes_1 + bytes_slope * double( bs - first_peer_bs );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus CompOp::add_output_bytes( uint16_t bs, uint32_t bytes ) {
  output_bytes_map[ bs ] = bytes;
  return ExitStatus::SUCCESS;
}

ExitStatus CompOp::add_comp_time( uint16_t bs, Step comp_time_steps ) {
  comp_time_map[ bs ] = comp_time_steps;
  return ExitStatus::SUCCESS;
}

ExitStatus NetOp::get_mem_size( uint64_t &size ) const {
  size = transfer_bytes;
  return ExitStatus::SUCCESS;
}

ExitStatus MemOp::get_mem_size( uint64_t &size ) const {
  size = num_bytes;
  return ExitStatus::SUCCESS;
}

ExitStatus CntrlOp::get_mem_size( uint64_t &size ) const {
  size = 0;
  return ExitStatus::SUCCESS;
}

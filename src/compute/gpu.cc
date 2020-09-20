#include <iostream>
#include "gpu.hh"

ExitStatus GPU::fetch_tx( Packet *&pkt ) {
  /* called by the interconnect only */
  if ( tx_buff.empty( )) {
    return ExitStatus::NOT_AVAILABLE;
  } else {
    pkt = tx_buff.front( );
    tx_buff.pop_front( );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::fill_rx( Packet *pkt, uint16_t num_pkts ) {
  /* called by the interconnect only */
  for ( uint16_t i = 0; i < num_pkts; i ++ ) {
    tp->recv( pkt ).ok( );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::packetize( NetOp *op ) {
  PacketId num_pkts = 0;
  double rem_byte = op->transfer_bytes;
  if ( rem_byte == 0 ) {
    std::cerr << "Zero-byte network transfer through interconnect: "
                 "try to route through pcie to improve efficiency." << std::endl;
    rem_byte = 1;
  }
  while ( rem_byte > 0 ) {
    uint16_t num_bytes = ( rem_byte > Packet::max_pkt_size ? Packet::max_pkt_size : rem_byte );
    Packet *p = new Packet( op->src_device, op->dst_device, num_bytes, curr_step /* tx_time */ );
    tp->tp_input.push_back( p );
    rem_byte -= num_bytes;
    num_pkts ++;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::compute( CompOp *op ) {
  assert( op->type == OpType::COMPUTE );
  assert( op->device->type == DeviceType::GPU );
  assert( op->device->dev_id == dev_id );
  op->start = ( curr_step < next_available ? next_available : curr_step );
  auto comp_time = op->comp_time;
  comp_time = ( comp_time > cnfg->gpu_min_comp_time ? comp_time : cnfg->gpu_min_comp_time );
  op->end = op->start + comp_time + cnfg->gpu_launch_latency;
  next_available = op->end;
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::communicate( NetOp *op ) {
  //Todo: implement pcie logic
  packetize( op ).ok( );
  PacketId last_pkt_id;
  tp->send( last_pkt_id ).ok( );
  recv_sig.emplace( last_pkt_id, op );
  op->start = Device::curr_step;
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::get_avail_memsize( uint64_t &size ) const {
  if ( total_memory < used_memory ) {
    return ExitStatus::FAILURE;
  }
  size = total_memory - used_memory;
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::allocate_mem( Op *op ) {
  uint64_t req_size;
  op->get_mem_size( req_size ).ok( );
  if ( req_size + used_memory > total_memory )
    return ExitStatus::FAILURE;
  used_memory += req_size;
  mem_map[ op ] = req_size;
  if ( op->type == OpType::COMPUTE ) {
    CompOp *comp_op;
    comp_op = static_cast<CompOp *>(op);
    load_est += comp_op->comp_time;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::deallocate_mem( Op *op ) {
  uint64_t req_size;
  op->get_mem_size( req_size ).ok( );
  used_memory -= req_size;
  mem_map.erase( op );
  if ( op->type == OpType::COMPUTE ) {
    CompOp *comp_op;
    comp_op = static_cast<CompOp *>(op);
    load_est -= comp_op->comp_time;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::check_mem_feas( Op *op, bool &is_feas ) {
  uint64_t req_size;
  op->get_mem_size( req_size ).ok( );
  is_feas = req_size + used_memory < total_memory;
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::release_mem_all( ) {
  mem_map.clear( );
  used_memory = 0;
  load_est = 0;
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::summary( ) const {
  std::cout << "GPU_" << dev_id << ":"
            << "used_memory=" << used_memory << " "
            << "total_memory=" << total_memory << " "
            << "load_est=" << load_est << " "
            << std::endl;
  return ExitStatus::SUCCESS;
}

ExitStatus GPU::setup_transport( int num_gpus ) {
  tp = new Transport( tx_buff, rx_buff, num_gpus );
  return ExitStatus::SUCCESS;
}

#include "base_placement.hh"

void device_utilization_stats( std::map< Device*, Step > load_dist_map, Step num_steps_est, bool verbose ){
  if (verbose){
    int i = 0;
    int j = 1;
    for ( auto d : load_dist_map ){
      if ( i == 0 ){
        std::cout <<  "r" << std::setfill('0') << std::setw(2) << j << ": ";
      }
      std::cout << std::setfill('0') << std::setw(2) << int( 100. * double( d.second ) / double( num_steps_est ) ) << ", ";
      i += 1;
      if ( i == 16 ){
        std::cout << std::endl;
        i = 0;
        j++;
      }
    }
    std::cout << std::endl;
  }
  double util_sum = 0;
  double n = 0;
  for ( auto d : load_dist_map ){
    double util = 100. * double( d.second ) / double( num_steps_est );
//    std::cout << util << " " <<  double( d.second ) << " " << double( num_steps_est ) << std::endl;
    util_sum += util;
    n ++;
  }
  std::cout << "avg_utilization=" << int( util_sum / n ) << "%."
            << std::endl;
}
ExitStatus BasePlacement::estimate_iter_time( CG &graph, Step &num_steps_est ) {
  /* get the bandwidth estimates */
  std::unordered_map< Device *, std::unordered_map< Device *, double>> bw_est;
  interconnect->offline_bw_est( bw_est ); /* bw_est will be in bytes per step */

  /* traverse the compute graph to find the total time */
  num_steps_est = 0;
  std::unordered_map< Device *, Step > comp_avail_time;
  std::unordered_map< Device *, Step > net_avail_time_tx;
  std::unordered_map< Device *, Step > net_avail_time_rx;

  std::map< uint32_t, Op * > run_order;
  graph.priority_sort( run_order );
  assert( graph.adj.size( ) == run_order.size( ));
  double comm_time = 0;
  Step path_avail_time;

  int n_netop = 0;
  uint32_t prior = 0;
  size_t count = 0;

  /* some informative stats */
  uint64_t total_xfer_bytes = 0;
  Step total_comp_load = 0;
  Step critical_path_load;
  int critical_path_len;
  graph.critical_path_load( critical_path_load );
  graph.critical_path_len( critical_path_len );
  std::map< Device*, Step > load_dist_map;

  while ( count < run_order.size( )) {
    if ( run_order.count( prior ) == 0 ) {
      prior ++;
      continue;
    }
    count ++;
    Op *op = run_order.at( prior );
//    std::cout << op->name << std::endl;
    prior ++;
    /* find when input dependencies are met */
    Step cand_start = 0;
    op->start = std::numeric_limits< Step >::max( );
    op->end = std::numeric_limits< Step >::max( );
    for ( auto pred : graph.reverse_adj.at( op ))
      cand_start = ( cand_start > pred->end ? cand_start : pred->end );

    /* handle operation time of each op node */
    if ( op->type == OpType::NETWORK ) {
      n_netop ++;
      auto *net_op = dynamic_cast<NetOp *>( op );
      if ( net_avail_time_tx.count( net_op->src_device ) == 0 )
        net_avail_time_tx[ net_op->src_device ] = 0;
      if ( net_avail_time_rx.count( net_op->dst_device ) == 0 )
        net_avail_time_rx[ net_op->dst_device ] = 0;
      path_avail_time = net_avail_time_tx.at( net_op->src_device ) > net_avail_time_rx.at( net_op->dst_device ) ?
          net_avail_time_tx.at( net_op->src_device ) : net_avail_time_rx.at( net_op->dst_device );
      op->start = ( cand_start > path_avail_time ? cand_start : path_avail_time );
      double out_degree = graph.adj.at( op ).size( );
      comm_time = net_op->transfer_bytes / bw_est[ net_op->src_device ][ net_op->dst_device ] * out_degree;
//      std::cout << net_op->transfer_bytes << " "
//                << bw_est[ net_op->src_device ][ net_op->dst_device ] << std::endl;
      comm_time =
          ( comm_time < interconnect->cnfg.interconnect_latency ? interconnect->cnfg.interconnect_latency : comm_time );
//      comm_time = comm_time < 10 ? 10 : comm_time; //todo: double check this
      op->end = op->start + Step( comm_time );
      total_xfer_bytes += net_op->transfer_bytes;
      net_avail_time_tx.at( net_op->src_device ) = op->end;
      net_avail_time_rx.at( net_op->dst_device ) = 0;//op->end;
    } else if ( op->type == OpType::MEMORY ) {
      op->start = cand_start;
      op->end = op->start;
    } else if ( op->type == OpType::COMPUTE && op->device->type == DeviceType::GPU ) {
      if ( comp_avail_time.count( op->device ) == 0 )
        comp_avail_time.emplace( op->device, 0 );
      auto *comp_op = dynamic_cast<CompOp *>( op );
      comp_op->start =
          ( cand_start > comp_avail_time.at( comp_op->device ) ? cand_start : comp_avail_time.at( comp_op->device ));
      auto comp_time = comp_op->comp_time;
      comp_time =
          ( comp_time < interconnect->cnfg.gpu_min_comp_time ? interconnect->cnfg.gpu_min_comp_time : comp_time );
      comp_op->end = comp_op->start + comp_time + interconnect->cnfg.gpu_launch_latency;
      comp_avail_time.at( op->device ) = comp_op->end;
      total_comp_load += comp_time;
      load_dist_map[ op->device ] += comp_time;
    } else if ( op->type == OpType::COMPUTE && op->device->type == DeviceType::CPU ) {
      auto *comp_op = dynamic_cast<CompOp *>( op );
      /* presume there are abundant cpu cores;
       * otherwise should a cpu available similar to gpus */
      comp_op->start = cand_start;
      comp_op->end = comp_op->start + comp_op->comp_time;
      total_comp_load += comp_op->comp_time;
      load_dist_map[ comp_op->device ] += comp_op->comp_time;
    } else if ( op->type == OpType::CONTROL_DEPENDENCY ) {
      op->start = cand_start;
      op->end = op->start;
    }
    /* num_steps_est equals the latest op finishing time in the graph */
    num_steps_est = ( num_steps_est > op->end ? num_steps_est : op->end );
  }
  std::cout << "total_xfer_bytes="    << total_xfer_bytes   * 1e-9 << "GB, "
            << "total_comp_load="     << total_comp_load    * 1e-3 << "ms, "
            << "critical_path_load="  << critical_path_load * 1e-3 << "ms, "
            << "critical_path_len="   << critical_path_len         << "."
            << std::endl;
  device_utilization_stats( load_dist_map, num_steps_est, false );


  return ExitStatus::SUCCESS;
}

struct device_less_than {
  inline bool operator()( const Op *op_1, const Op *op_2 ) {
    return ( op_1->device->dev_id < op_2->device->dev_id );
  }
};

ExitStatus BasePlacement::add_ring_reduce( CG &graph, std::vector< Op * > replicas ) {
  if ( replicas.size( ) == 1 )
    return ExitStatus::SUCCESS;
  /* sort device ids */
  std::sort( replicas.begin( ), replicas.end( ), device_less_than( ));
  Op *net_op;
  MemOp *src_op;
  MemOp *dst_op;
  CntrlOp *cntrl_op;
  cntrl_op = new CntrlOp( "RingReduce_Control_" + replicas[ 0 ]->name,
                          OpType::CONTROL_DEPENDENCY,
                          &gpus[ 0 ], //ToDo: set the device to cpu
                          replicas[ 0 ]->session_id,
                          "c8" );
  for ( unsigned long i = 0; i < replicas.size( ); i ++ ) {
    src_op = dynamic_cast< MemOp * >( replicas[ i ] );
    dst_op = dynamic_cast< MemOp * >( replicas[ ( i + 1 ) % replicas.size( ) ] );
    if ( src_op->device->dev_id != dst_op->device->dev_id ) {
      net_op = new NetOp( "RingReduce_" + src_op->name + "_" + dst_op->name,
                          OpType::NETWORK,
                          interconnect,
                          src_op->session_id,
                          src_op->num_bytes,
                          src_op->device,
                          dst_op->device,
                          "c9" );
      graph.add_edge( src_op, net_op );
      graph.add_edge( net_op, cntrl_op );
    } else {
      graph.add_edge( src_op, cntrl_op );
      graph.add_edge( dst_op, cntrl_op );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BasePlacement::add_dp_ring_reduce( CG &graph, std::vector< Op * > replicas, int dp_degree ) {
  if ( replicas.size( ) == 1 )
    return ExitStatus::SUCCESS;
  /* sort device ids */
  std::sort( replicas.begin( ), replicas.end( ), device_less_than( ));
  Op *net_op;
  MemOp *src_op;
  MemOp *dst_op;
  CntrlOp *cntrl_op;
  cntrl_op = new CntrlOp( "RingReduce_Control_" + replicas[ 0 ]->name,
                          OpType::CONTROL_DEPENDENCY,
                          &gpus[ 0 ], //ToDo: set the device to cpu
                          replicas[ 0 ]->session_id,
                          "c8" );
  for ( unsigned long i = 0; i < replicas.size( ); i ++ ) {
    src_op = dynamic_cast< MemOp * >( replicas[ i ] );
    dst_op = dynamic_cast< MemOp * >( replicas[ ( i + 1 ) % replicas.size( ) ] );
    if ( src_op->device->dev_id != dst_op->device->dev_id ) {
      int dst_dev_id = ( dst_op->device->dev_id + dp_degree ) % interconnect->num_gpus;
      GPU* dst_dev = &( interconnect->gpus[ dst_dev_id ] );
      net_op = new NetOp( "RingReduce_" + src_op->name + "_" + dst_op->name,
                          OpType::NETWORK,
                          interconnect,
                          src_op->session_id,
                          src_op->num_bytes,
                          src_op->device,
                          dst_dev,
                          "c9" );
      graph.add_edge( src_op, net_op );
      graph.add_edge( net_op, cntrl_op );
    } else {
      graph.add_edge( src_op, cntrl_op );
      graph.add_edge( dst_op, cntrl_op );
    }
  }
  return ExitStatus::SUCCESS;
}

uint16_t ring_dist( uint16_t a, uint16_t b, uint16_t ring_size ) {
  /* number of hops from a to b  */
  int dist = ( b >= a ? b - a : ring_size - ( a - b ));
  return dist;
}

ExitStatus BasePlacement::add_async_netops( CG &output_graph ) {
  NetOp *net_op;
  Op *src_op;
  uint16_t src_batch;
  uint16_t dst_batch;

  for ( auto e : output_graph.adj ) {
    src_op = e.first;
    for ( auto dst_op : e.second ) {
      if ( src_op->device != dst_op->device &&
          src_op->device->type == DeviceType::GPU &&
          dst_op->device->type == DeviceType::GPU &&
          ( src_op->type == OpType::COMPUTE || src_op->type == OpType::MEMORY ) &&
          ( dst_op->type == OpType::COMPUTE || dst_op->type == OpType::MEMORY )) {
        uint32_t num_transfer_bytes;
        if ( src_op->type == OpType::COMPUTE && dst_op->type == OpType::COMPUTE ) {
          /* only send the needed samples in the batch */
          static_cast<CompOp *>(src_op)->get_batch_size( src_batch ).ok( );
          static_cast<CompOp *>(dst_op)->get_batch_size( dst_batch ).ok( );
//          std::cout << "src_batch=" << src_batch << " "
//                    << "dst_batch=" << dst_batch << " "
//                    << std::endl;
          double ratio = double( dst_batch ) / double( src_batch );
//          std::cout << "src_batch=" << src_batch << " "
//                    << "dst_batch=" << dst_batch << " "
//                    << "ratio="     << ratio     << " "
//                    << std::endl;
          num_transfer_bytes = static_cast<CompOp *>(src_op)->output_bytes;
          num_transfer_bytes = ratio * double( num_transfer_bytes ); /* just the send required samples */
//          std::cout << num_transfer_bytes << std::endl;
        } else if ( src_op->type == OpType::COMPUTE ) {
          num_transfer_bytes = static_cast<CompOp *>(src_op)->output_bytes;
        } else if ( src_op->type == OpType::MEMORY ) {
          num_transfer_bytes = static_cast<MemOp *>(src_op)->num_bytes;
        } else {
          throw std::runtime_error( "Didn't expect this type of communication" );
        }
        if ( num_transfer_bytes > 0 ) { //todo: route the others through pcie
          net_op = new NetOp( "Memcpy_" + src_op->name + "_" + dst_op->name,
                              OpType::NETWORK,
                              interconnect,
                              src_op->session_id,
                              num_transfer_bytes,
                              src_op->device,
                              dst_op->device,
                              "c1" );
          output_graph.add_edge( src_op, net_op );
          output_graph.add_edge( net_op, dst_op );
        }
      }
    }
  }
  return ExitStatus::SUCCESS;
}

//ExitStatus BasePlacement::add_sync_netops( CG &output_graph, std::unordered_map< Op *, std::vector< Op *>> &replicas ) {
//  MemOp *mem_op;
//  for ( auto rop : replicas ) {
//    if ( rop.first->type == OpType::MEMORY ) {
//      mem_op = dynamic_cast< MemOp * >( rop.first );
//      if ( mem_op->mem_type == MemType::WRITEVARIABLE ) {
//        add_ring_reduce( output_graph, rop.second ).ok( );
//      }
//    }
//  }
//  return ExitStatus::SUCCESS;
//}


ExitStatus BasePlacement::add_sync_netops( CG &output_graph, std::map< Op *, std::vector< Op *>> &replicas ) {
  MemOp *mem_op;
  for ( auto rop : replicas ) {
    if ( rop.first->type == OpType::MEMORY ) {
      mem_op = dynamic_cast< MemOp * >( rop.first );
      if ( mem_op->mem_type == MemType::WRITEVARIABLE ) {
        add_ring_reduce( output_graph, rop.second ).ok( );
      }
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus BasePlacement::add_dp_sync_netops( CG &output_graph, std::map< Op *, std::vector< Op *>> &replicas, int dp_degree ) {
  MemOp *mem_op;
  for ( auto rop : replicas ) {
    if ( rop.first->type == OpType::MEMORY ) {
      mem_op = dynamic_cast< MemOp * >( rop.first );
      if ( mem_op->mem_type == MemType::WRITEVARIABLE ) {
        add_dp_ring_reduce( output_graph, rop.second, dp_degree ).ok( );
      }
    }
  }
  return ExitStatus::SUCCESS;
}


#include "mp.hh"

uint16_t ring_distance( uint16_t a, uint16_t b, uint16_t ring_size ) {
  /* number of hops from a to b  */
  int dist = ( b >= a ? b - a : ring_size - ( a - b ));
  return dist;
}

ExitStatus MP::get_earliest_available( int &ready_dev_id,
                                       const vector< Step > &v,
                                       const uint32_t start_offset,
                                       const uint32_t end_offset,
                                       uint64_t mem_size ) {
    //todo: doublecheck the uint32_t of offsets; it used to be int :)
  uint32_t start = start_offset % v.size( );
  //start = ( start < 0 ? start + v.size( ) : start );
  uint32_t end;
  if ( end_offset - start_offset > v.size( )) {
    end = 1 + ( start_offset + v.size( ) - 1 ) % v.size( );
  } else {
    end = 1 + ( end_offset - 1 ) % v.size( );
  }
  ready_dev_id = 0;
  uint64_t avail_mem;
  if ( start >= end ) {
    for ( size_t i = start; i < v.size( ); i ++ ) {
      gpus[ i ].get_avail_memsize( avail_mem ).ok( );
      if ( v.at( i ) < v.at( ready_dev_id ) && avail_mem > mem_size )
        ready_dev_id = i;
    }
    for ( size_t i = 0; i < end; i ++ ) {
      gpus[ i ].get_avail_memsize( avail_mem ).ok( );
      if ( v.at( i ) < v.at( ready_dev_id ) && avail_mem > mem_size )
        ready_dev_id = i;
    }
  } else {
    for ( size_t i = start; i < end; i ++ ) {
      gpus[ i ].get_avail_memsize( avail_mem ).ok( );
      if ( v.at( i ) < v.at( ready_dev_id ) && avail_mem > mem_size ) {
        ready_dev_id = i;
      }
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus MP::place_partitions_heuristic( CG &output_graph ) {
  /* make sure all gpu stats are reset */
  for ( size_t i = 0; i < num_gpus; i ++ ) {
    gpus[ i ].release_mem_all( );
  }
  stack< Op * > rev_sorted_graph;
  stack< Op * > sorted_graph;
  input_graph.topological_sort( sorted_graph );
  /* assign op priorities */
  Op *main_op;
  uint32_t rep_priority = 0;
  uint32_t main_priority = 0;
  while ( ! sorted_graph.empty( )) {
    main_op = sorted_graph.top( );
    main_op->priority = main_priority;
    main_priority ++;
    for ( auto op : parallel_ops_map.at( main_op )) {
      op->priority = rep_priority;
      rep_priority ++;
    }
    rev_sorted_graph.push( main_op );
    sorted_graph.pop( );
  }
  while ( ! rev_sorted_graph.empty( )) {
    main_op = rev_sorted_graph.top( );
    sorted_graph.push( main_op );
    rev_sorted_graph.pop( );
  }
  Step iter_time = 0;
  /* keep a track of when each device is available */
  vector< Step > comp_avail_time;
  map< Op *, Step > start;
  map< Op *, Step > end;
  for ( int i = 0; i < avail_gpus; i ++ ) {
    comp_avail_time.push_back( Step( 0 ));
  }
  while ( ! sorted_graph.empty( )) {
    main_op = sorted_graph.top( );
    for ( auto op : parallel_ops_map.at( main_op )) {
      auto preds = output_graph.reverse_adj.at( op );
      int ready_dev_id;
      /* place the op itself if it's not already placed */
      if ( op->device == nullptr ) {
        /* find the first available device */
        Step cand_start = 0;
        start[ op ] = std::numeric_limits< Step >::max( );
        end[ op ] = std::numeric_limits< Step >::max( );
        uint16_t start_id_min = avail_gpus;
        uint16_t start_id_max = 0;
        for ( auto pred : preds ) {
          cand_start = ( cand_start > end.at( pred ) ? cand_start : end.at( pred ));
          start_id_max = ( pred->device->dev_id > start_id_max ? pred->device->dev_id : start_id_max );
          start_id_min = ( pred->device->dev_id < start_id_min ? pred->device->dev_id : start_id_min );
        }
        /* set the leftmost as the start_id */
        uint16_t start_id = ( start_id_max - start_id_min > start_id_min + avail_gpus - start_id_max ?
            start_id_max : start_id_min );
        uint16_t end_id = ( start_id_max - start_id_min > start_id_min + avail_gpus - start_id_max ?
            start_id_min : start_id_max );
        uint16_t range_lo = end_id;
        uint16_t range_hi = start_id + d_max;
        if ( ring_distance( range_lo, range_hi, avail_gpus ) > d_max ) {
          range_hi = end_id;
        }
        uint64_t mem_size;
        op->get_mem_size( mem_size ).ok( );
        get_earliest_available( ready_dev_id, comp_avail_time, range_lo, range_hi, mem_size );
        op->device = &gpus[ ready_dev_id ];
        if ( gpus[ ready_dev_id ].allocate_mem( op ) != ExitStatus::SUCCESS ) {
          return ExitStatus::NOT_AVAILABLE;
        }
        start[ op ] = ( cand_start > comp_avail_time[ ready_dev_id ] ? cand_start : comp_avail_time[ ready_dev_id ] );
        Step duration;
        if ( op->type == OpType::COMPUTE ) {
          duration = dynamic_cast<CompOp *>(op)->comp_time;
          duration =
              ( duration < interconnect->cnfg.gpu_min_comp_time ? interconnect->cnfg.gpu_min_comp_time : duration );
          duration += interconnect->cnfg.gpu_launch_latency;
        } else if ( op->type == OpType::MEMORY ) {
          duration = 1;
        } else {
          duration = 0;
        }
        end[ op ] = start[ op ] + duration;
        iter_time = ( iter_time < end[ op ] ? end[ op ] : iter_time );
        comp_avail_time[ ready_dev_id ] = end[ op ];
      }
    }
    rev_sorted_graph.push( main_op );
    sorted_graph.pop( );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus MP::find_placement( CG &output_graph ) {
  if ( place_partitions_heuristic( output_graph ) != ExitStatus::SUCCESS )
    return ExitStatus::NOT_AVAILABLE;

  /* add network ops */
  add_async_netops( output_graph ).ok( );

  /* add syncronization ops */
//  add_sync_netops( output_graph, replicas ).ok( );

  /* add mat-parallel ops */
//  add_global_dp_ops( output_graph ).ok( );

  /* fix all the priorities */
  set< Op * > priorless_ops;
  for ( auto e : output_graph.adj ) {
    if ( e.first->priority == 0 ) {
      priorless_ops.emplace( e.first );
    }
  }
  std::map< uint32_t, Op * > run_order;
  output_graph.priority_sort( run_order );
  vector< Op * > ordered_ops;
  while ( output_graph.adj.size( ) != run_order.size( )) {
    set< Op * > visited;
    for ( uint32_t prior = 0; prior < run_order.size( ); prior ++ ) {
      Op *prior_op = run_order.at( prior );
      for ( auto pred : output_graph.reverse_adj.at( prior_op )) {
        if ( visited.count( pred ) == 0 ) {
          visited.emplace( pred );
          ordered_ops.push_back( pred );
        }
        visited.emplace( prior_op );
        ordered_ops.push_back( prior_op );
      }
    }
    uint32_t new_prior = 0;
    for ( auto op : ordered_ops ) {
      op->priority = new_prior;
      new_prior ++;
    }
    assert( new_prior == ordered_ops.size( ));
    run_order.clear( );
    output_graph.priority_sort( run_order );
  }
  assert( output_graph.adj.size( ) == run_order.size( ));
  return ExitStatus::SUCCESS;
}

ExitStatus MP::num_batch_splits( Op *, uint32_t & ) {
  return ExitStatus::FAILURE;
}

ExitStatus MP::add_global_dp_ops( CG &output_graph ) {
  //todo: deprecate
  uint32_t dp_degree = interconnect->num_gpus / avail_gpus;
  uint32_t mp_degree = avail_gpus;
  if ( dp_degree == 1 )
    return ExitStatus::SUCCESS;
  map< Op *, Op * > cntrl_ops_map;
  map< Op *, vector< Op * > > global_mem_ops_map;
  for ( auto e : output_graph.adj ) {
    if ( e.first->type == OpType::MEMORY ) {
      MemOp *mem_op;
      mem_op = static_cast< MemOp * >( e.first );
      if ( mem_op->mem_type == MemType::WRITEVARIABLE ) {
        /* first add a control op to make sure the variable is ready */
        CntrlOp *cntrl_op;
        cntrl_op = new CntrlOp( "DP_Control_" + mem_op->name,
                                OpType::CONTROL_DEPENDENCY,
                                &gpus[ 0 ], //ToDo: set the device to cpu
                                mem_op->session_id,
                                "global_dp_1" );
        cntrl_ops_map[ mem_op ] = cntrl_op;
        MemOp *global_mem_op;
        for ( size_t r = 0; r < dp_degree; r ++ ) {
          uint32_t dev_id = ( r * mp_degree + mem_op->device->dev_id ) % num_gpus;
          global_mem_op = new MemOp( "DP_Global_" + mem_op->name,
                                     OpType::MEMORY,
                                     &gpus[ dev_id ],
                                     mem_op->session_id,
                                     MemType::WRITEVARIABLE,
                                     mem_op->num_bytes,
                                     "global_dp_2" );
          global_mem_ops_map[ mem_op ].push_back( global_mem_op );
        }
      }
    }
  }
  for ( auto p : cntrl_ops_map ) {
    Op *op = p.first;
    output_graph.add_edge( op, p.second );
    for ( auto mem_op : global_mem_ops_map.at( op )) {
      output_graph.add_edge( p.second, mem_op );
    }
    add_ring_reduce( output_graph, global_mem_ops_map.at( op ));
  }
  return ExitStatus::SUCCESS;
}


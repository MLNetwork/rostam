#include "op_partitioner.hh"

ExitStatus OpPartitioner::split_compute( CompOp *op, uint32_t &num_splits ) {
  int raw_splits = quant_step > 0 ? double( op->comp_time ) / double( quant_step ) : max_splits.at( op );
  raw_splits = ( raw_splits < 1 ? 1 : raw_splits );
  /* make it a power of 2 */
  uint16_t pow = log2( raw_splits );
  num_splits = ( 1 << pow );
  num_splits = ( num_splits > max_splits.at( op ) ? max_splits.at( op ) : num_splits );
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitioner::split_memory( MemOp *op,
                                        const map< Op *, uint32_t > &comp_splits_map,
                                        uint32_t &num_splits ) {
  num_splits = 1;
  switch ( op->mem_type ) {
    case MemType::READVARIABLE:
      for ( auto succ : input_graph.adj.at( op )) {
        if ( succ->type == OpType::COMPUTE ) {
          num_splits = comp_splits_map.at( succ );
          break;
        }
      }
      break;
    case MemType::WRITEVARIABLE:
      for ( auto pred : input_graph.reverse_adj.at( op )) {
        if ( pred->type == OpType::COMPUTE ) {
          num_splits = comp_splits_map.at( pred );
          break;
        }
      }
      break;
    default:throw runtime_error( "Not implemented." );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitioner::get_nsplits_all( map< Op *, uint32_t > &splits_map ) {
  stack< Op * > sorted_graph;
  stack< Op * > rev_sorted_graph;
  input_graph.topological_sort( sorted_graph );
  /* first divide the compute ops */
  uint32_t num_splits;
  Op *op;
  while ( ! sorted_graph.empty( )) {
    op = sorted_graph.top( );
    if ( op->type == OpType::COMPUTE ) {
      split_compute( dynamic_cast<CompOp *>(op), num_splits ).ok( );
      splits_map[ op ] = num_splits;
    } else {
      rev_sorted_graph.push( op );
    }
    sorted_graph.pop( );
  }

  /* now that all compute ops are partionted, 
   * split the rest according to the compute ops */
  while ( ! rev_sorted_graph.empty( )) {
    op = rev_sorted_graph.top( );
    if ( op->type == OpType::MEMORY ) {
      split_memory( dynamic_cast<MemOp *>(op), splits_map, num_splits ).ok( );
      splits_map[ op ] = num_splits;
    } else if ( op->type == OpType::CONTROL_DEPENDENCY ) {
      splits_map[ op ] = 1;
    } else
      throw runtime_error( "This type of op not predicted." );
    rev_sorted_graph.pop( );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitioner::partition( CG &output_graph, map< Op *, vector< Op *>> &parallel_ops_map ) {
  map< Op *, uint32_t > splits_map;
  get_nsplits_all( splits_map ).ok( );
  create_parallel_ops( splits_map, parallel_ops_map ).ok( );
  add_data_dependencies( parallel_ops_map, output_graph );
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitionerSample::create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                                     map< Op *, vector< Op *>> &parallel_ops_map ) {
  for ( auto &e : input_graph.adj ) {
    Op *op = e.first;
    uint32_t num_splits = splits_map.at( op );
    if ( num_splits == 0 )
      cerr << "Zero-size partitioning";
    uint32_t num_bytes;
    uint16_t old_bs;
    uint16_t new_bs;
    switch ( op->type ) {
      case OpType::COMPUTE:CompOp *comp_op;
        comp_op = dynamic_cast< CompOp * >( op );
        comp_op->get_batch_size( old_bs );
        new_bs = old_bs / num_splits;
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          CompOp *rep_op;
          rep_op = new CompOp( comp_op->name + "_" + std::to_string( i ),
                               OpType::COMPUTE,
                               comp_op->device,
                               comp_op->session_id,
                               "batch_create_parallel_ops",
                               comp_op->comp_time_map,
                               comp_op->output_bytes_map );
          static_cast<CompOp *>(rep_op)->set_batch_size( new_bs );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        assert( parallel_ops_map[ op ].size( ) == num_splits );
        break;
      case OpType::MEMORY:MemOp *mem_op;
        mem_op = dynamic_cast< MemOp * >( op );
        num_bytes = mem_op->num_bytes;
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          Op *rep_op;
          rep_op = new MemOp( mem_op->name + "_" + std::to_string( i ),
                              OpType::MEMORY,
                              mem_op->device,
                              mem_op->session_id,
                              mem_op->mem_type,
                              num_bytes,
                              "batch_create_parallel_ops" );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        break;
      case OpType::CONTROL_DEPENDENCY:CntrlOp *cntrl_op;
        cntrl_op = dynamic_cast< CntrlOp * >( op );
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          Op *rep_op;
          rep_op = new CntrlOp( cntrl_op->name + "_" + std::to_string( i ),
                                OpType::CONTROL_DEPENDENCY,
                                cntrl_op->device,
                                cntrl_op->session_id,
                                "batch_create_parallel_ops" );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        break;
      default: throw std::runtime_error( "Did not expect this!" );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitionerSample::add_data_dependencies( const map< Op *, vector< Op *>> &parallel_ops_map,
                                                       CG &output_graph ) {
  for ( auto &e : input_graph.adj ) {
    Op *src = e.first;
    for ( auto dst : e.second ) {
      auto src_size = parallel_ops_map.at( src ).size( );
      auto dst_size = parallel_ops_map.at( dst ).size( );
      if ( src_size >= dst_size ) {
        uint32_t r;
        r = src_size / dst_size;
        for ( uint32_t i = 0; i < dst_size; i ++ ) {
          for ( uint32_t j = 0; j < r; j ++ ) {
            output_graph.add_edge( parallel_ops_map.at( src )[ i * r + j ],
                                   parallel_ops_map.at( dst )[ i ] );
          }
        }
      } else {
        uint32_t r;
        r = dst_size / src_size;
        for ( uint32_t i = 0; i < src_size; i ++ ) {
          for ( uint32_t j = 0; j < r; j ++ ) {
            output_graph.add_edge( parallel_ops_map.at( src )[ i ],
                                   parallel_ops_map.at( dst )[ i * r + j ] );
          }
        }
      }
    }
  }
  return ExitStatus::SUCCESS;
}

OpPartitionerSample::OpPartitionerSample( Step quant_step,
                                          const map< Op *, uint32_t > &max_splits,
                                          const CG &input_graph ) : OpPartitioner(
    quant_step,
    max_splits,
    input_graph ) { }

ExitStatus OpPartitionerParam::create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                                    map< Op *, vector< Op *>> &parallel_ops_map ) {
  for ( auto &e : input_graph.adj ) {
    Op *op = e.first;
    uint32_t num_splits = splits_map.at( op );
    if ( num_splits == 0 )
      cerr << "Zero-size partitioning";
    uint32_t num_bytes;
    switch ( op->type ) {
      case OpType::COMPUTE:CompOp *comp_op;
        comp_op = dynamic_cast< CompOp * >( op );
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          CompOp *rep_op;
          rep_op = new CompOp( comp_op->name + "_" + std::to_string( i ),
                               OpType::COMPUTE,
                               comp_op->device,
                               comp_op->session_id,
                               "param_create_parallel_ops" + std::to_string( num_splits ),
                               comp_op->comp_time_map,
                               comp_op->output_bytes_map );
          comp_op->copy_scale_to( rep_op, 1./ double( num_splits ));
          parallel_ops_map[ op ].push_back( rep_op );
        }
        assert( parallel_ops_map[ op ].size( ) == num_splits );
        break;
      case OpType::MEMORY:MemOp *mem_op;
        mem_op = dynamic_cast< MemOp * >( op );
        num_bytes = mem_op->num_bytes;
        num_bytes = num_bytes / num_splits;
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          Op *rep_op;
          rep_op = new MemOp( mem_op->name + "_" + std::to_string( i ),
                              OpType::MEMORY,
                              mem_op->device,
                              mem_op->session_id,
                              mem_op->mem_type,
                              num_bytes,
                              "param_create_parallel_ops" );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        break;
      case OpType::CONTROL_DEPENDENCY:CntrlOp *cntrl_op;
        cntrl_op = dynamic_cast< CntrlOp * >( op );
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          Op *rep_op;
          rep_op = new CntrlOp( cntrl_op->name + "_" + std::to_string( i ),
                                OpType::CONTROL_DEPENDENCY,
                                cntrl_op->device,
                                cntrl_op->session_id,
                                "param_create_parallel_ops" );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        break;
      default: throw std::runtime_error( "Did not expect this!" );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitionerParam::add_data_dependencies( const map< Op *, vector< Op *>> &parallel_ops_map,
                                                      CG &output_graph ) {
  for ( auto &e : input_graph.adj ) {
    Op *src = e.first;
    for ( auto dst : e.second ) {
      auto src_size = parallel_ops_map.at( src ).size( );
      auto dst_size = parallel_ops_map.at( dst ).size( );
      for ( uint32_t i = 0; i < src_size; i ++ ) {
        for ( uint32_t j = 0; j < dst_size; j ++ ) {
          if ( src->type == OpType::COMPUTE && dst->type == OpType::COMPUTE )
            output_graph.add_edge( parallel_ops_map.at( src )[ i ],
                                   parallel_ops_map.at( dst )[ j ] );
        }
      }
      if ( src_size >= dst_size ) {
        uint32_t r;
        r = src_size / dst_size;
        for ( uint32_t i = 0; i < dst_size; i ++ ) {
          for ( uint32_t j = 0; j < r; j ++ ) {
            if ( src->type != OpType::COMPUTE || dst->type != OpType::COMPUTE )
              output_graph.add_edge( parallel_ops_map.at( src )[ i * r + j ],
                                     parallel_ops_map.at( dst )[ i ] );
          }
        }
      } else if ( src_size < dst_size ) {
        uint32_t r;
        r = dst_size / src_size;
        for ( uint32_t i = 0; i < src_size; i ++ ) {
          for ( uint32_t j = 0; j < r; j ++ ) {
            if ( src->type != OpType::COMPUTE || dst->type != OpType::COMPUTE )
              output_graph.add_edge( parallel_ops_map.at( src )[ i ],
                                     parallel_ops_map.at( dst )[ i * r + j ] );
          }
        }
      }
    }
  }
  return ExitStatus::SUCCESS;
}

OpPartitionerParam::OpPartitionerParam( Step quant_step,
                                        const map< Op *, uint32_t > &max_splits,
                                        const CG &input_graph ) : OpPartitioner(
    quant_step,
    max_splits,
    input_graph ) { }

OpPartitionerAttribute::OpPartitionerAttribute( Step quant_step,
                                                const map< Op *, uint32_t > &max_splits,
                                                const CG &input_graph ) : OpPartitioner( quant_step,
                                                                                         max_splits,
                                                                                         input_graph ) { }

ExitStatus OpPartitionerAttribute::create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                                        map< Op *, vector< Op *>> &parallel_ops_map ) {
  for ( auto &e : input_graph.adj ) {
    Op *op = e.first;
    uint32_t num_splits = splits_map.at( op );
//    cout << num_splits << endl;
    if ( num_splits == 0 )
      cerr << "Zero-size partitioning";
    uint32_t num_bytes;
    switch ( op->type ) {
      case OpType::COMPUTE:CompOp *comp_op;
        comp_op = dynamic_cast< CompOp * >( op );
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          CompOp *rep_op;
          rep_op = new CompOp( comp_op->name + "_" + std::to_string( i ),
                               OpType::COMPUTE,
                               comp_op->device,
                               comp_op->session_id,
                               "param_create_parallel_ops" + std::to_string( num_splits ),
                               comp_op->comp_time_map,
                               comp_op->output_bytes_map );
          /* 1. use the same batch size as the original op */
          uint16_t bs;
          comp_op->get_batch_size( bs );
          rep_op->set_batch_size( bs );
          /* 2. overwrite the estimated time and memory */
          comp_op->copy_scale_to( rep_op, 1. / double( num_splits ));

          parallel_ops_map[ op ].push_back( rep_op );
        }
        assert( parallel_ops_map[ op ].size( ) == num_splits );
        break;
      case OpType::MEMORY:MemOp *mem_op;
        mem_op = dynamic_cast< MemOp * >( op );
        num_bytes = mem_op->num_bytes;
        num_bytes = num_bytes / num_splits;
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          Op *rep_op;
          rep_op = new MemOp( mem_op->name + "_" + std::to_string( i ),
                              OpType::MEMORY,
                              mem_op->device,
                              mem_op->session_id,
                              mem_op->mem_type,
                              num_bytes,
                              "param_create_parallel_ops" );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        break;
      case OpType::CONTROL_DEPENDENCY:CntrlOp *cntrl_op;
        cntrl_op = dynamic_cast< CntrlOp * >( op );
        for ( uint32_t i = 0; i < num_splits; i ++ ) {
          Op *rep_op;
          rep_op = new CntrlOp( cntrl_op->name + "_" + std::to_string( i ),
                                OpType::CONTROL_DEPENDENCY,
                                cntrl_op->device,
                                cntrl_op->session_id,
                                "param_create_parallel_ops" );
          parallel_ops_map[ op ].push_back( rep_op );
        }
        break;
      default: throw std::runtime_error( "Did not expect this!" );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus OpPartitionerAttribute::add_data_dependencies( const map< Op *, vector< Op *>> &parallel_ops_map,
                                                          CG &output_graph ) {
  for ( auto &e : input_graph.adj ) {
    Op *src = e.first;
    for ( auto dst : e.second ) {
      auto src_size = parallel_ops_map.at( src ).size( );
      auto dst_size = parallel_ops_map.at( dst ).size( );
      if ( src_size >= dst_size ) {
        uint32_t r;
        r = src_size / dst_size;
        for ( uint32_t i = 0; i < dst_size; i ++ ) {
          for ( uint32_t j = 0; j < r; j ++ ) {
            output_graph.add_edge( parallel_ops_map.at( src )[ i * r + j ],
                                   parallel_ops_map.at( dst )[ i ] );
          }
        }
      } else {
        uint32_t r;
        r = dst_size / src_size;
        for ( uint32_t i = 0; i < src_size; i ++ ) {
          for ( uint32_t j = 0; j < r; j ++ ) {
            output_graph.add_edge( parallel_ops_map.at( src )[ i ],
                                   parallel_ops_map.at( dst )[ i * r + j ] );
          }
        }
      }
    }
  }
  return ExitStatus::SUCCESS;
}

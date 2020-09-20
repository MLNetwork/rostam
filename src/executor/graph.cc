#include "graph.hh"

using namespace std;

ExitStatus CG::summary( ) {
  uint64_t size;
  uint64_t total_mem_size = 0;
  for ( auto n : adj ) {
    n.first->get_mem_size( size );
    total_mem_size += size;
  }
  cout << "total_mem_size=" << total_mem_size << endl;
  int max_depth;
  critical_path_len( max_depth );
  cout << "critical computational path length: " << max_depth << endl;
  return ExitStatus::SUCCESS;
}

ExitStatus CG::summary( const string log_dir ) {
  std::ofstream log_file = std::ofstream( log_dir + "graph.log", std::ofstream::out );

  uint64_t size;
  uint64_t total_mem_size = 0;
  for ( auto n : adj ) {
    n.first->get_mem_size( size );
    total_mem_size += size;
    if ( n.first->type == OpType::COMPUTE )
      log_file << static_cast<CompOp *>(n.first)->comp_time << " ";
  }
  cout << "total_mem_size=" << total_mem_size << endl;
  int max_depth;
  critical_path_len( max_depth );
  cout << "critical computational path length: " << max_depth << endl;
  return ExitStatus::SUCCESS;
}

ExitStatus CG::from_graph_profile( std::string filename, const double step_size_sec, const int num_profiles ) {
  /* Verify that the version of the library that we linked against is
   * compatible with the version of the headers we compiled against. */
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  graph_profile::Profile profile;

  /* store the ops profile mat */
  map< string, Op * > op_list;
  CompOp *cn;
  MemOp *mn;
  CntrlOp *nopn;
  double step_ratio = 1e-6 / step_size_sec;
  Step comp_time_steps;
  MemType mem_type = MemType::INVALID;
  uint16_t bs = 1;
  for ( int prof_no = 0; prof_no < num_profiles; prof_no ++ ) {
    // Read the profile from the file.
    fstream input( filename + "_bs" + to_string( bs ) + ".pb", ios::in | ios::binary );
    if ( ! profile.ParseFromIstream( &input )) {
      cerr << "Failed to parse the input graph profile." << endl;
      return ExitStatus::FAILURE;
    }
    for ( int i = 0; i < profile.nodes_size( ); i ++ ) {
      const graph_profile::Profile::Op &op = profile.nodes( i );
      switch ( op.op_type( )) {
        case graph_profile::Profile::Op::OpType::Profile_Op_OpType_COMPUTE:
          comp_time_steps = Step( double( op.comp_time_us( )) * step_ratio );
          uint32_t output_bytes;
          output_bytes = op.output_bytes( );

          if ( op_list.count( op.name( )) == 0 ) {
            std::map< uint16_t, Step > comp_time_map;
            std::map< uint16_t, uint32_t > output_bytes_map;
            comp_time_map[ bs ] = comp_time_steps;
            output_bytes_map[ bs ] = output_bytes;
            cn = new CompOp( op.name( ), OpType::COMPUTE, nullptr, 0, "c2", comp_time_map, output_bytes_map );
          } else {
            CompOp *old_op;
            old_op = dynamic_cast<CompOp *>( op_list.at( op.name( )));
            old_op->add_comp_time( bs, comp_time_steps );
            old_op->add_output_bytes( bs, output_bytes );
            if ( i == profile.nodes_size( ) - 1 ) {
              /* to speed-up extrapolations */
              auto extra_bs = bs;
              auto extra_comp_time_steps = comp_time_steps;
              auto extra_output_bytes = output_bytes;
              for ( int k = 0; k < 5; k ++ ) {
                extra_bs *= 2;
                extra_comp_time_steps *= 2;
                extra_output_bytes *= 2;
                old_op->add_comp_time( bs, extra_comp_time_steps );
                old_op->add_output_bytes( bs, extra_output_bytes );
              }
            }
          }
          op_list.emplace( op.name( ), cn );
          break;
        case graph_profile::Profile::Op::OpType::Profile_Op_OpType_MEMORY:
          switch ( op.mem_type( )) {
            case graph_profile::Profile::Op::MemType::Profile_Op_MemType_CONSTANT:throw runtime_error( "CONSTANT MemType is deprecated. Please use READVARIABLE/WRITEVARIABLE." );
              break;
            case graph_profile::Profile::Op::MemType::Profile_Op_MemType_READVARIABLE:mem_type = MemType::READVARIABLE;
              break;
            case graph_profile::Profile::Op::MemType::Profile_Op_MemType_WRITEVARIABLE:
              mem_type = MemType::WRITEVARIABLE;
              break;
            case graph_profile::Profile::Op::MemType::Profile_Op_MemType_TENSOR:throw runtime_error( "Tensor MemType is deprecated. Please use READVARIABLE/WRITEVARIABLE." );
              break;
            case graph_profile::Profile::Op::MemType::Profile_Op_MemType_Profile_Op_MemType_INT_MAX_SENTINEL_DO_NOT_USE_:
              throw runtime_error( "Not expected INT_MAX_SENTINEL_DO_NOT_USE_" );
            case graph_profile::Profile::Op::MemType::Profile_Op_MemType_Profile_Op_MemType_INT_MIN_SENTINEL_DO_NOT_USE_:
              throw runtime_error( "Not expected INT_MIN_SENTINEL_DO_NOT_USE_" );
          }
          mn = new MemOp( op.name( ), OpType::MEMORY, nullptr, 0, mem_type, op.num_bytes( ), "c3" );
          op_list.emplace( op.name( ), mn );
          break;
        case graph_profile::Profile::Op::OpType::Profile_Op_OpType_CONTROLDEPENDENCY:
          nopn = new CntrlOp( op.name( ), OpType::CONTROL_DEPENDENCY, nullptr, 0, "c4" );
          op_list.emplace( op.name( ), nopn );
          break;
        default: throw runtime_error( "Cannot parse all op types." );
      }
    }
    bs = bs << 1;
  }

  /* add graph structure */
  for ( int i = 0; i < profile.graph_size( ); i ++ ) {
    const graph_profile::Profile::Adjacents &adjacents = profile.graph( i );
    const string &src = profile.graph( i ).node( );
    for ( int j = 0; j < adjacents.succs_size( ); j ++ ) {
      const string &dst = profile.graph( i ).succs( j );
      add_edge( op_list.at( src ), op_list.at( dst ));
    }
  }

  /* Delete all global objects allocated by libprotobuf. */
  google::protobuf::ShutdownProtobufLibrary( );
  return ExitStatus::SUCCESS;
}

ExitStatus CG::release_ops( ) {
  for ( auto op : adj ) {
    delete op.first;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus CG::scale_graph( CG &scaled_graph, const double &batch_factor ) const {
  std::map< Op *, Op * > old_to_new;
  for ( auto e : adj ) {
    CompOp *new_comp_op;
    CompOp *old_comp_op;
    MemOp *new_mem_op;
    MemOp *old_mem_op;
    CntrlOp *new_cntrl_op;
    CntrlOp *old_cntrl_op;
    switch ( e.first->type ) {
      case OpType::COMPUTE:old_comp_op = static_cast<CompOp *>( e.first );
        new_comp_op = new CompOp( );
        old_comp_op->copy_scale_to( new_comp_op, batch_factor );
        old_to_new[ old_comp_op ] = new_comp_op;
        break;
      case OpType::MEMORY:old_mem_op = static_cast<MemOp *>( e.first );
        new_mem_op = new MemOp( *old_mem_op );
        old_to_new[ old_mem_op ] = new_mem_op;
        break;
      case OpType::CONTROL_DEPENDENCY:old_cntrl_op = static_cast<CntrlOp *>( e.first );
        new_cntrl_op = new CntrlOp( *old_cntrl_op );
        old_to_new[ old_cntrl_op ] = new_cntrl_op;
        break;
      default :throw runtime_error( "Not implemented." );
    }
  }
  for ( auto e : adj ) {
    for ( auto succ : e.second ) {
      scaled_graph.add_edge( old_to_new.at( e.first ), old_to_new.at( succ ));
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus CG::priority_sort( std::map< uint32_t, Op * > &prior_sorted ) {
  for ( auto e : adj ) {
    prior_sorted[ e.first->priority ] = e.first;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus CG::critical_path_len( int &max_depth ) {
  max_depth = 0;
  std::stack< Op * > stack;
  topological_sort( stack );
  map< Op *, int > depth_map;
  while ( ! stack.empty( )) {
    Op *op = stack.top( );
    int depth = 0;
    for ( auto pred : reverse_adj.at( op )) {
      depth = ( depth > depth_map.at( pred ) ? depth : depth_map.at( pred ));
    }
    if ( op->type == OpType::COMPUTE && dynamic_cast<CompOp *>( op )->comp_time > 0 ) {
      depth ++;
    }
    depth_map[ op ] = depth;
    if ( depth > max_depth ) {
      max_depth = depth;
    }
    stack.pop( );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus CG::set_global_batchsize( uint16_t bs ) const {
  for ( auto e : adj ) {
    Op *op = e.first;
    if ( op->type == OpType::COMPUTE ) {
      static_cast<CompOp *>( op )->set_batch_size( bs ).ok( );
    }
  }
  return ExitStatus::SUCCESS;
}

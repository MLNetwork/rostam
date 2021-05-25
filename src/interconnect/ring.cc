#include "ring.hh"
#include "graph.hh"

ExitStatus RingInterconnect::setup_ilp_solver( ) {
#ifdef HAVE_GUROBI

  Matrix2D< double > normal_tm( eff_num_gpus, eff_num_gpus );
  normalize_tm( normal_tm );

  /* Create an environment */
  env = new GRBEnv( );

  /* Create an empty model */
  env->start( );
  model = new GRBModel( *env );
  model->set( GRB_IntParam_OutputFlag, 0 );
  /* create the wavelength allocation integer decisions: lambda[ eff_num_gpus, eff_num_gpus ] */
  int num_directions = 2;
  lambda = new GRBVar ***[num_directions];
  for ( int dir_no = 0; dir_no < num_directions; dir_no ++ ) {
    lambda[ dir_no ] = new GRBVar **[num_rings];
    for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
      lambda[ dir_no ][ ring_no ] = new GRBVar *[eff_num_gpus];
      for ( int g1 = 0; g1 < eff_num_gpus; g1 ++ ) {
        lambda[ dir_no ][ ring_no ][ g1 ] = new GRBVar[eff_num_gpus];
        for ( int g2 = 0; g2 < eff_num_gpus; g2 ++ ) {
          lambda[ dir_no ][ ring_no ][ g1 ][ g2 ] = model->addVar( 0.0,
                                                                   num_waves / num_rings / 2,
                                                                   0.0,
                                                                   GRB_INTEGER,
                                                                   "lambda_" + to_string( dir_no ) + "_"
                                                                       + to_string( ring_no ) + "_" + to_string( g1 )
                                                                       + "_"
                                                                       + to_string( g2 ));
        }
      }
    }
  }

  /* wavelength no conflict constraints */
  GRBLinExpr ***segments; /* num_directions x num_rings x eff_num_gpus */
  segments = new GRBLinExpr** [num_directions];
  for ( int dir_no = 0; dir_no < num_directions; dir_no++ ){
    segments[dir_no] = new GRBLinExpr* [num_rings];
    for ( int ring_no = 0; ring_no < num_rings; ring_no++ ){
      segments[dir_no][ring_no] = new GRBLinExpr[eff_num_gpus];
      }
    }
//  for ( int seg_no = 0; seg_no < eff_num_gpus; seg_no ++ ) {
//    segments[ seg_no ] = 0;
//  }
  for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
    for ( int src = 0; src < eff_num_gpus; src ++ ) {
      for ( int dst = src; dst < src + eff_num_gpus; dst ++ ) {
        for ( int seg_no = src; seg_no < dst; seg_no ++ ) {
          segments[ 0 ][ ring_no ][ seg_no % eff_num_gpus ] += lambda[ 0 ][ ring_no ][ src ][ dst % eff_num_gpus ];
        }
      }
    }
  }
  for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
    for ( int src = 0; src < eff_num_gpus; src ++ ) {
      for ( int dst = src; dst < src + eff_num_gpus; dst ++ ) {
        for ( int seg_no = src; seg_no < dst; seg_no ++ ) {
          segments[ 1 ][ ring_no ][ seg_no % eff_num_gpus ] += lambda[ 1 ][ ring_no ][ dst % eff_num_gpus ][ src ];
        }
      }
    }
  }

  /* wavelengths should not share any segments */
  for ( int dir_no = 0; dir_no < num_directions; dir_no ++ ) {
    for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
      for ( int seg_no = 0; seg_no < eff_num_gpus; seg_no ++ ) {
        model->addConstr( segments[ dir_no ][ ring_no ][ seg_no ],
                          GRB_LESS_EQUAL,
                          num_waves / num_rings / 2,
                          "segment_constraint_" + to_string( dir_no ) + "_" + to_string( ring_no ) + "_"
                              + to_string( seg_no ));
      }
    }
  }


  /* Obviously, nodes do not use precious wavelengths to
   * talk to their own self :D (to speed-up the solver) */
  for ( int dir_no = 0; dir_no < num_directions; dir_no ++ ) {
    for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
      for ( int g = 0; g < eff_num_gpus; g ++ ) {
        model->addConstr( lambda[ dir_no ][ ring_no ][ g ][ g ],
                          GRB_EQUAL,
                          0.0,
                          "politeness_constraint_" + to_string( dir_no ) + "_" + to_string( ring_no ) + "_"
                              + to_string( g ));
      }
    }
  }
  GRBLinExpr** bw; /* eff_num_gpus x eff_num_gpus */
  bw = new GRBLinExpr*[eff_num_gpus];
  for ( int src = 0; src < eff_num_gpus; src ++ ) {
    bw[src] = new GRBLinExpr[eff_num_gpus];
  }
  for ( int dir_no = 0; dir_no < num_directions; dir_no ++ ) {
    for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
      for ( int src = 0; src < eff_num_gpus; src ++ ) {
        for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
          bw[ src ][ dst ] += lambda[ dir_no ][ ring_no ][ src ][ dst ];
        }
      }
    }
  }
  /* going to maximize min_throughput */
  GRBVar
      min_throughput = model->addVar( 0.0, GRB_INFINITY, 1.0 /* obj_func_coeff */, GRB_CONTINUOUS, "min_throughput" );
  for ( int src = 0; src < eff_num_gpus; src ++ ) {
    for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
      if ( normal_tm.get_elem( src, dst ) > 0 ) {
        model->addConstr( min_throughput,
                          GRB_LESS_EQUAL,
                          bw[ src ][ dst ] / normal_tm.get_elem( src, dst ),
                          "throughput_constraint_" + to_string( src ) + "_" + to_string( dst ));
      }
    }
  }
  /* set objective */
  model->set( GRB_IntAttr_ModelSense, GRB_MAXIMIZE );
#endif //HAVE_GUROBI
  return ExitStatus::SUCCESS;
}

ExitStatus RingInterconnect::allocate_episode_bw_ilp( ) {
#ifdef HAVE_GUROBI

  Matrix2D< double > normal_tm( eff_num_gpus, eff_num_gpus );
  normalize_tm( normal_tm );
//  cout << normal_tm << endl;
  model->reset( ); /* reset solution states */
  model->update( );
  /* first, try to remove any previous constraint related to the traffic matrix */
  for ( int src = 0; src < eff_num_gpus; src ++ ) {
    for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
      string s = "throughput_constraint_" + to_string( src ) + "_" + to_string( dst );
      try {
        model->remove( model->getConstrByName( s ));
      } catch ( GRBException &e ) {
        if ( e.getErrorCode( ) != 10003 ) {
          cout << "Error code = " << e.getErrorCode( ) << endl;
          cout << e.getMessage( ) << endl;
        }
      } catch ( ... ) {
        cout << "Exception during optimization" << endl;
      }
    }
  }
  model->update( ); /* gurobi is lazy :D */
  /* then, add the new traffic constraints */
  int num_directions = 2;
  GRBLinExpr** bw; /* eff_num_gpus x eff_num_gpus */
  bw = new GRBLinExpr*[eff_num_gpus];
  for ( int src = 0; src < eff_num_gpus; src ++ ) {
    bw[src] = new GRBLinExpr[eff_num_gpus];
  }
  for ( int dir_no = 0; dir_no < num_directions; dir_no ++ ) {
    for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
      for ( int src = 0; src < eff_num_gpus; src ++ ) {
        for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
          bw[ src ][ dst ] += lambda[ dir_no ][ ring_no ][ src ][ dst ];
        }
      }
    }
  }
  for ( int src = 0; src < eff_num_gpus; src ++ ) {
    for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
      string s = "throughput_constraint_" + to_string( src ) + to_string( dst );
      if ( normal_tm.get_elem( src, dst ) > 0 ) {
        model->addConstr( model->getVarByName( "min_throughput" ),
                          GRB_LESS_EQUAL,
                          bw[ src ][ dst ] / normal_tm.get_elem( src, dst ),
                          "throughput_constraint_" + to_string( src ) + "_" + to_string( dst ));
      }
    }
  }
  model->getEnv().set(GRB_DoubleParam_TimeLimit, 10);
  model->update( );
  model->optimize( );
  episode_bw.fill_zeros( );

  for ( int dir_no = 0; dir_no < num_directions; dir_no ++ ) {
    for ( int ring_no = 0; ring_no < num_rings; ring_no ++ ) {
      for ( int src = 0; src < eff_num_gpus; src ++ ) {
        for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
          episode_bw.add_elem_by( src,
                                  dst,
                                  ( model->getVarByName(
                                      "lambda_" + to_string( dir_no ) + "_" + to_string( ring_no ) + "_"
                                          + to_string( src ) +
                                          "_" + to_string( dst )).get( GRB_DoubleAttr_X ) )
                                      * double( cnfg.bwxstep_per_wave ));
        }
      }
    }
  }
  for ( int src = 0; src < eff_num_gpus; src ++ ) {
    for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
      double bitrate = episode_bw.get_elem(src, dst);
      bool is_connected = ( bitrate > 0);
      if ( is_connected ) {
        sparse_episode_bw[ src ][ dst ] = bitrate;
      }
      }
  }
//  for ( int src = 0; src < eff_num_gpus; src ++ ) {
//    for ( int dst = 0; dst < eff_num_gpus; dst ++ ) {
//      double bitrate = episode_bw.get_elem( src, dst );
//      cout << bitrate << " ";
//    }
//    cout << endl;
//  }
//  cout << "==========================" << endl;
#endif //HAVE_GUROBI
  return ExitStatus::SUCCESS;
}

ExitStatus RingInterconnect::allocate_episode_bw_mcf( ) {
#ifdef HAVE_GUROBI

  Matrix2D< double > normal_tm( eff_num_gpus, eff_num_gpus );
  normalize_tm( normal_tm );
  try {
    GRBModel mcf_model = GRBModel( GRBEnv( ));
    mcf_model.set( GRB_IntParam_OutputFlag, 0 );
    const double capacity = 1.0;
    uint16_t src;
    uint16_t dst;
    using EdgeWeight = tuple< uint16_t, uint16_t, double >;
    vector< EdgeWeight > flow_weights;
    Graph< uint16_t > flow_graph;
    for ( src = 0; src < eff_num_gpus; src ++ ) {
      for ( dst = 0; dst < eff_num_gpus; dst ++ ) {
        if ( normal_tm.get_elem( src, dst ) > 0 ) {
          flow_weights.emplace_back( src, dst, - 1.0 / normal_tm.get_elem( src, dst ));
          flow_graph.add_edge( src, dst );
        } else if ( dst == ( src + 1 ) % eff_num_gpus ) {
          flow_weights.emplace_back( src, dst, 0 ); /* dummy edge weight */
          flow_graph.add_edge( src, dst );
        }
      }
    }
    GRBVar *flows;
    flows = new GRBVar[flow_weights.size( )];
    for ( size_t i = 0; i < flow_weights.size( ); i ++ ) {
      string var_name = "flow_" +
          to_string( get< 0 >( flow_weights[ i ] )) + "to" +
          to_string( get< 1 >( flow_weights[ i ] ));
      flows[ i ] = mcf_model.addVar( 0.0, capacity, get< 2 >( flow_weights[ i ] ), GRB_CONTINUOUS, var_name );
    }
    mcf_model.update( );

    /* flow conservation */
    for ( auto &flow_weight : flow_weights ) {
      int node = get< 0 >( flow_weight );
      GRBLinExpr input_flow = 0;
      for ( auto pred : flow_graph.reverse_adj.at( node )) {
        string fin_name = "flow_" + to_string( pred ) + "to" + to_string( node );
        input_flow += mcf_model.getVarByName( fin_name );
      }
      GRBLinExpr output_flow = 0;
      for ( auto succ : flow_graph.adj.at( node )) {
        string fout_name = "flow_" + to_string( node ) + "to" + to_string( succ );
        output_flow += mcf_model.getVarByName( fout_name );
      }
      mcf_model.addConstr( input_flow, GRB_EQUAL, output_flow, "flow_conservation_constraint_" + to_string( node ));
    }

    /* capacity constraint:
     * the total flow that can pass through each segment is bounded by
     * that segment's capacity */
    for ( int seg_no = 0; seg_no < eff_num_gpus; seg_no ++ ) {
      GRBLinExpr total_flow = 0;
      for ( auto &flow_weight : flow_weights ) {
        src = get< 0 >( flow_weight );
        dst = get< 1 >( flow_weight );
        int seg_offset = ( seg_no - int( src ));
        seg_offset = ( seg_offset < 0 ? eff_num_gpus - seg_offset : seg_offset );
        int dst_offset = ( int( dst ) - int( src ));
        dst_offset = ( dst_offset < 0 ? eff_num_gpus - dst_offset : dst_offset );
        if ( dst_offset > seg_offset ) {
          total_flow += mcf_model.getVarByName( "flow_" + to_string( src ) + "to" + to_string( dst ));
        }
      }
      mcf_model.addConstr( total_flow, GRB_LESS_EQUAL, capacity, "capacity_constraint_" + to_string( seg_no ));
    }

    /* set objective */
    mcf_model.set( GRB_IntAttr_ModelSense, GRB_MINIMIZE );

    /* solve */
    mcf_model.optimize( );

    /* rounding */
    Matrix2D< uint16_t > allocation( eff_num_gpus, eff_num_gpus );
    double wave_inv = 1.0 / double( num_waves );
    for ( auto &flow_weight : flow_weights ) {
      src = get< 0 >( flow_weight );
      dst = get< 1 >( flow_weight );
      double alloc = mcf_model.getVarByName( "flow_" + to_string( src ) + "to" + to_string( dst ))
          .get( GRB_DoubleAttr_X ) / wave_inv;
      auto rounded_alloc = uint16_t( alloc );
      double diff = alloc - rounded_alloc;
      double r = (( double ) rand( ) / ( RAND_MAX ));
      alloc = ( r > diff ? alloc : alloc + 1 );
      allocation.set_elem( src, dst, alloc );
    }

    /* handle rounding errors */
    for ( int seg_no = 0; seg_no < eff_num_gpus; seg_no ++ ) {
      int total_waves = 0;
      for ( auto &flow_weight : flow_weights ) {
        src = get< 0 >( flow_weight );
        dst = get< 1 >( flow_weight );
        int seg_offset = ( seg_no - int( src ));
        seg_offset = ( seg_offset < 0 ? eff_num_gpus - seg_offset : seg_offset );
        int dst_offset = ( int( dst ) - int( src ));
        dst_offset = ( dst_offset < 0 ? eff_num_gpus - dst_offset : dst_offset );
        if ( dst_offset > seg_offset ) {
          total_waves += allocation.get_elem( src, dst );
          /* if exceeding the total number of available waves,
           * deallocate waves until the constraint is met */
          while ( total_waves > num_waves ) {
            allocation.sub_elem_by( src, dst, 1 );
            total_waves -= 1;
          }
        }
      }
    }

    episode_bw.mul_by( cnfg.bwxstep_per_wave );
    delete[] flows;
  } catch ( GRBException &e ) {
    cout << "Error code = " << e.getErrorCode( ) << endl;
    cout << e.getMessage( ) << endl;
  } catch ( ... ) {
    cout << "Exception during optimization" << endl;
  }
#endif //HAVE_GUROBI
  return ExitStatus::SUCCESS;
}

ExitStatus RingInterconnect::allocate_episode_bw( ) {
#ifdef HAVE_GUROBI

  switch ( bw_decision_type ) {
    case BWDecisionType::ILP:allocate_episode_bw_ilp( ).ok( );
      break;
    case BWDecisionType::MINCOSTFLOW:allocate_episode_bw_mcf( ).ok( );
      break;
    default:throw runtime_error( "Not implemented this type of bandwidth decsion making for RingInterconnect." );
  }
#endif //HAVE_GUROBI
  return ExitStatus::SUCCESS;
}

ExitStatus RingInterconnect::offline_bw_est( std::unordered_map< Device *,
                                                                 std::unordered_map< Device *, double>> &estimate ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      estimate[ &gpus[ i ]][ &gpus[ j ]] = cnfg.num_waves * cnfg.bwxstep_per_wave;
    }
  }


  return ExitStatus::SUCCESS;
}

RingInterconnect::RingInterconnect( uint16_t dev_id,
                                    GPU *gpus,
                                    uint16_t num_gpus,
                                    double ingress_link_speed,
                                    double egress_link_speed,
                                    TMEstimatorBase *tm_estimator,
                                    const SimConfig &cnfg,
                                    const uint16_t num_waves,
                                    const BWDecisionType bw_decision_type,
                                    const int tolerable_dist,
                                    const int num_rings,
                                    const std::string log_dir ) : BaseInterconnect( dev_id,
                                                                                    gpus,
                                                                                    num_gpus,
                                                                                    ingress_link_speed,
                                                                                    egress_link_speed,
                                                                                    tm_estimator,
                                                                                    cnfg,
                                                                                    log_dir ),
#ifdef HAVE_GUROBI
                                                                  env( nullptr ),
                                                                  model( nullptr ),
                                                                  lambda( nullptr ),
#endif //HAVE_GUROBI
                                                                  num_waves( num_waves ),
                                                                  bw_decision_type( bw_decision_type ),
                                                                  tolerable_dist( tolerable_dist ),
                                                                  num_rings( num_rings ) {

}

RingInterconnect::~RingInterconnect( ) {
}

ExitStatus RingInterconnect::reset_routing_step_counters( ){
  for ( auto &s : sparse_episode_bw ){
    for ( auto &d : s.second ){
      sparse_episode_bw_budget[ s.first ][ d.first ] = d.second;
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus RingInterconnect::is_routing_feasible( Packet* pkt, bool &is_bw_avail ){
  if ( curr_step % ( cnfg.dec_interval + cnfg.interconnect_reconf_delay ) < cnfg.interconnect_reconf_delay ) {
    /* we are still in interconnect transition mode; no packet transfer is feasible */
    is_bw_avail = false;
    return ExitStatus::SUCCESS;
  }

  try {
    is_bw_avail = ( sparse_episode_bw_budget.at( pkt->src->dev_id ).at( pkt->dst->dev_id ) >= pkt->num_bytes );
    if ( is_bw_avail )
      sparse_episode_bw_budget.at( pkt->src->dev_id ).at( pkt->dst->dev_id ) -= pkt->num_bytes;
  }
  catch ( const std::out_of_range& oor ) {
    is_bw_avail = false;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus RingInterconnect::set_eff_num_gpus( uint16_t n ) {
  eff_num_gpus = n;
  return ExitStatus::SUCCESS;
}

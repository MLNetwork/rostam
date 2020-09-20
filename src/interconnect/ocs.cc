#include "ocs.hh"

ExitStatus OCSInterconnect::allocate_episode_bw( ) {
  if ( single_shot ){
  allocate_episode_bw_singleshot( ).ok( );
  }
  else
    allocate_episode_bw_multishot().ok();
  return ExitStatus::SUCCESS;
}

ExitStatus OCSInterconnect::setup_optimal_solver( ) {
  if ( single_shot ){
  setup_optimal_solver_singleshot( ).ok( );
  }
  else
    setup_optimal_solver_multishot().ok();
  return ExitStatus::SUCCESS;
}

ExitStatus OCSInterconnect::setup_optimal_solver_singleshot( ) {
#ifdef HAVE_GUROBI

  try {
    Matrix2D< double > normal_tm( num_gpus, num_gpus );
    normalize_tm( normal_tm );
    model->set( GRB_IntParam_OutputFlag, 0 );
    /* create the permutation decisions */
    GRBVar ***perms; /* a num_ocs x port_count x port_count binary variable */
    perms = new GRBVar **[num_ocs];
    for ( int i = 0; i < num_ocs; i ++ ) {
      perms[ i ] = new GRBVar *[port_count];
      for ( int j = 0; j < port_count; j ++ ) {
        perms[ i ][ j ] = new GRBVar[port_count];
        for ( int k = 0; k < port_count; k ++ ) {
          perms[ i ][ j ][ k ] = model->addVar( 0.0,
                                                1.0,
                                                0.0,
                                                GRB_BINARY,
                                                "perm_" + to_string( i ) +
                                                    "_" + to_string( j ) +
                                                    "_" + to_string( k ));
        }
      }
    }

    /* permutation row constraints */
    for ( int sw = 0; sw < num_ocs; sw ++ ) {
      for ( int src = 0; src < port_count; src ++ ) {
        GRBLinExpr expr = 0;
        for ( int dst = 0; dst < port_count; dst ++ ) {
          expr += perms[ sw ][ src ][ dst ];
        }
        string s = "egress_constraint_sw" + to_string( sw ) + "_port" + to_string( src );
        model->addConstr( expr, GRB_EQUAL, 1.0, s );
      }
    }

    /* permutation column constraints */
    for ( int sw = 0; sw < num_ocs; sw ++ ) {
      for ( int dst = 0; dst < port_count; dst ++ ) {
        GRBLinExpr expr = 0;
        for ( int src = 0; src < port_count; src ++ ) {
          expr += perms[ sw ][ src ][ dst ];
        }
        string s = "ingress_constraint_sw" + to_string( sw ) + "_port" + to_string( dst );
        model->addConstr( expr, GRB_EQUAL, 1.0, s );
      }
    }

    /* traffic completion time */
    GRBVar min_rate = model->addVar( 0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "min_rate" );
    /* set objective */
    model->setObjective( GRBLinExpr( min_rate ), GRB_MAXIMIZE );

    /* create device-to-device bandwidths */
    GRBLinExpr **bw;
    bw = new GRBLinExpr *[num_gpus];
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      bw[ src_dev ] = new GRBLinExpr[num_gpus];
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        bw[ src_dev ][ dst_dev ] = 0;
      }
    }
    for ( int src_port = 0; src_port < port_count; src_port ++ ) {
      for ( int dst_port = 0; dst_port < port_count; dst_port ++ ) {
        for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
          uint16_t src_dev = port_map.at( ocs_no ).at( src_port )->dev_id;
          uint16_t dst_dev = port_map.at( ocs_no ).at( dst_port )->dev_id;
          bw[ src_dev ][ dst_dev ] += perms[ ocs_no ][ src_port ][ dst_port ];
        }
      }
    }
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        if ( normal_tm.get_elem( src_dev, dst_dev ) > 0 ) {
          GRBLinExpr rate = bw[ src_dev ][ dst_dev ] / normal_tm.get_elem( src_dev, dst_dev );
          string s = "rate_constraint_port" + to_string( src_dev ) + "_port" + to_string( dst_dev );
          model->addConstr( rate, GRB_GREATER_EQUAL, min_rate, s );
        }
      }
    }
  }
  catch ( GRBException e ) {
    if ( e.getErrorCode( ) != 10003 ) {
      cout << "Error code = " << e.getErrorCode( ) << endl;
      cout << e.getMessage( ) << endl;
    }
  }
#endif
  return ExitStatus::SUCCESS;
}

ExitStatus OCSInterconnect::allocate_episode_bw_singleshot( ) {
#ifdef HAVE_GUROBI
  try {
    Matrix2D< double > normal_tm( num_gpus, num_gpus );
    normalize_tm( normal_tm );
    model->reset( ); /* reset solution states */
    /* create device-to-device bandwidths */
    GRBLinExpr **bw;
    bw = new GRBLinExpr *[num_gpus];
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      bw[ src_dev ] = new GRBLinExpr[num_gpus];
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        bw[ src_dev ][ dst_dev ] = 0;
      }
    }
    for ( int src_port = 0; src_port < port_count; src_port ++ ) {
      for ( int dst_port = 0; dst_port < port_count; dst_port ++ ) {
        for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
          uint16_t src_dev = port_map.at( ocs_no ).at( src_port )->dev_id;
          uint16_t dst_dev = port_map.at( ocs_no ).at( dst_port )->dev_id;
          bw[ src_dev ][ dst_dev ] += model->getVarByName( "perm_" + to_string( ocs_no ) +
              "_" + to_string( src_port ) +
              "_" + to_string( dst_port ));
        }
      }
    }

    /* first remove the old constraint */
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        string s = "rate_constraint_port" + to_string( src_dev ) + "_port" + to_string( dst_dev );
        try {
          model->remove( model->getConstrByName( s ));
        }
        catch ( GRBException e ) {
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
    /* then, add the new constraint */
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        string s = "rate_constraint_port" + to_string( src_dev ) + "_port" + to_string( dst_dev );
        if ( normal_tm.get_elem( src_dev, dst_dev ) > 0 ) {
          GRBLinExpr rate = bw[ src_dev ][ dst_dev ] / normal_tm.get_elem( src_dev, dst_dev );
          model->addConstr( rate, GRB_GREATER_EQUAL, model->getVarByName( "min_rate" ), s );
        }
      }
    }

    model->update( );
    model->optimize( );
    episode_bw.fill_zeros( );
    for ( int src_port = 0; src_port < port_count; src_port ++ ) {
      for ( int dst_port = 0; dst_port < port_count; dst_port ++ ) {
        for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
          uint16_t src_dev = port_map.at( ocs_no ).at( src_port )->dev_id;
          uint16_t dst_dev = port_map.at( ocs_no ).at( dst_port )->dev_id;
          episode_bw.add_elem_by( src_dev, dst_dev, model->getVarByName( "perm_" + to_string( ocs_no ) +
              "_" + to_string( src_port ) +
              "_" + to_string( dst_port )).get( GRB_DoubleAttr_X ));
        }
      }
    }

    episode_bw.mul_by( double( cnfg.num_waves ) * double( cnfg.bwxstep_per_wave ) / double( num_ocs ));
  }
  catch ( GRBException e ) {
    if ( e.getErrorCode( ) != 10003 ) {
      cout << "Error code = " << e.getErrorCode( ) << endl;
      cout << e.getMessage( ) << endl;
    }
  }
  #endif //HAVE_GUROBI

  return ExitStatus::SUCCESS;
}

ExitStatus OCSInterconnect::setup_optimal_solver_multishot( ) {
#ifdef HAVE_GUROBI

  try {
    Matrix2D< double > normal_tm( num_gpus, num_gpus );
    normalize_tm( normal_tm );
    model->set( GRB_IntParam_OutputFlag, 0 );
    /* create the permutation decisions */
    GRBVar ***perms; /* a num_ocs x port_count x port_count binary variable */
    perms = new GRBVar **[num_ocs];
    for ( int i = 0; i < num_ocs; i ++ ) {
      perms[ i ] = new GRBVar *[port_count];
      for ( int j = 0; j < port_count; j ++ ) {
        perms[ i ][ j ] = new GRBVar[port_count];
        for ( int k = 0; k < port_count; k ++ ) {
          perms[ i ][ j ][ k ] = model->addVar( 0.0,
                                                1.0,
                                                0.0,
                                                GRB_BINARY,
                                                "perm_" + to_string( i ) +
                                                    "_" + to_string( j ) +
                                                    "_" + to_string( k ));
        }
      }
    }

    /* permutation row constraints */
    for ( int sw = 0; sw < num_ocs; sw ++ ) {
      for ( int src = 0; src < port_count; src ++ ) {
        GRBLinExpr expr = 0;
        for ( int dst = 0; dst < port_count; dst ++ ) {
          expr += perms[ sw ][ src ][ dst ];
        }
        string s = "egress_constraint_sw" + to_string( sw ) + "_port" + to_string( src );
        model->addConstr( expr, GRB_EQUAL, 1.0, s );
      }
    }

    /* permutation column constraints */
    for ( int sw = 0; sw < num_ocs; sw ++ ) {
      for ( int dst = 0; dst < port_count; dst ++ ) {
        GRBLinExpr expr = 0;
        for ( int src = 0; src < port_count; src ++ ) {
          expr += perms[ sw ][ src ][ dst ];
        }
        string s = "ingress_constraint_sw" + to_string( sw ) + "_port" + to_string( dst );
        model->addConstr( expr, GRB_EQUAL, 1.0, s );
      }
    }

    /* create device-to-device bandwidths */
    GRBLinExpr **bw;
    bw = new GRBLinExpr *[num_gpus];
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      bw[ src_dev ] = new GRBLinExpr[num_gpus];
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        bw[ src_dev ][ dst_dev ] = 0;
      }
    }
    for ( int src_port = 0; src_port < port_count; src_port ++ ) {
      for ( int dst_port = 0; dst_port < port_count; dst_port ++ ) {
        for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
          uint16_t src_dev = port_map.at( ocs_no ).at( src_port )->dev_id;
          uint16_t dst_dev = port_map.at( ocs_no ).at( dst_port )->dev_id;
          bw[ src_dev ][ dst_dev ] += perms[ ocs_no ][ src_port ][ dst_port ];
        }
      }
    }
    GRBLinExpr rate;
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        if ( normal_tm.get_elem( src_dev, dst_dev ) > 0 ) {
          double alpha = normal_tm.get_elem( src_dev, dst_dev );
          rate += ( bw[ src_dev ][ dst_dev ] * alpha );
        }
      }
    }
    /* set objective */
    model->setObjective( rate, GRB_MAXIMIZE );
  }
  catch ( GRBException e ) {
    if ( e.getErrorCode( ) != 10003 ) {
      cout << "Error code = " << e.getErrorCode( ) << endl;
      cout << e.getMessage( ) << endl;
    }
  }
#endif
  return ExitStatus::SUCCESS;
}

ExitStatus OCSInterconnect::allocate_episode_bw_multishot( ) {
#ifdef HAVE_GUROBI

  try {
    Matrix2D< double > normal_tm( num_gpus, num_gpus );
    normalize_tm( normal_tm );
    model->reset( ); /* reset solution states */
    /* create device-to-device bandwidths */
    GRBLinExpr **bw;
    bw = new GRBLinExpr *[num_gpus];
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      bw[ src_dev ] = new GRBLinExpr[num_gpus];
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        bw[ src_dev ][ dst_dev ] = 0;
      }
    }
    for ( int src_port = 0; src_port < port_count; src_port ++ ) {
      for ( int dst_port = 0; dst_port < port_count; dst_port ++ ) {
        for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
          uint16_t src_dev = port_map.at( ocs_no ).at( src_port )->dev_id;
          uint16_t dst_dev = port_map.at( ocs_no ).at( dst_port )->dev_id;
          bw[ src_dev ][ dst_dev ] += model->getVarByName( "perm_" + to_string( ocs_no ) +
              "_" + to_string( src_port ) +
              "_" + to_string( dst_port ));
        }
      }
    }

    model->update( );
    GRBLinExpr rate;
    for ( int src_dev = 0; src_dev < num_gpus; src_dev ++ ) {
      for ( int dst_dev = 0; dst_dev < num_gpus; dst_dev ++ ) {
        if ( normal_tm.get_elem( src_dev, dst_dev ) > 0 ) {
          double alpha = normal_tm.get_elem( src_dev, dst_dev );
          rate += ( bw[ src_dev ][ dst_dev ] * alpha );
        }
      }
    }
    /* set objective */
    model->setObjective( rate, GRB_MAXIMIZE );

    model->update( );
    model->optimize( );
    episode_bw.fill_zeros( );
    for ( int src_port = 0; src_port < port_count; src_port ++ ) {
      for ( int dst_port = 0; dst_port < port_count; dst_port ++ ) {
        for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
          uint16_t src_dev = port_map.at( ocs_no ).at( src_port )->dev_id;
          uint16_t dst_dev = port_map.at( ocs_no ).at( dst_port )->dev_id;
          episode_bw.add_elem_by( src_dev, dst_dev, model->getVarByName( "perm_" + to_string( ocs_no ) +
              "_" + to_string( src_port ) +
              "_" + to_string( dst_port )).get( GRB_DoubleAttr_X ));
        }
      }
    }

    episode_bw.mul_by( double( cnfg.num_waves ) * double( cnfg.bwxstep_per_wave ) / double( num_ocs ));
  }
  catch ( GRBException e ) {
    if ( e.getErrorCode( ) != 10003 ) {
      cout << "Error code = " << e.getErrorCode( ) << endl;
      cout << e.getMessage( ) << endl;
    }
  }
#endif
  return ExitStatus::SUCCESS;
}

ExitStatus OCSInterconnect::offline_bw_est( unordered_map< Device *, unordered_map< Device *, double>> &estimate ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      estimate[ &gpus[ i ]][ &gpus[ j ]] = cnfg.num_waves * cnfg.bwxstep_per_wave;
      if (( i > num_ocs || j > num_ocs ))//&& single_shot
        estimate[ &gpus[ i ]][ &gpus[ j ]] = 1; /* something very small */
    }
  }
  return ExitStatus::SUCCESS;
}



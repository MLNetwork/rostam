#include "mordia.hh"

ExitStatus MordiaInterconnect::setup_ilp_solver( ) {
#ifdef HAVE_GUROBI
  Matrix2D< double > normal_tm( num_gpus, num_gpus );
  normalize_tm( normal_tm );

  /* Create an environment */
  env = new GRBEnv( );

  /* Create an empty model */
  env->start( );
  model = new GRBModel( *env );
  model->set( GRB_IntParam_OutputFlag, 0 );
  /* create the wavelength allocation binary decisions: lambda[ num_waves, num_gpus, num_gpus ] */
  lambda = new GRBVar **[num_waves];
  for ( int w = 0; w < num_waves; w ++ ) {
    lambda[ w ] = new GRBVar *[num_gpus];
    for ( int g1 = 0; g1 < num_gpus; g1 ++ ) {
      lambda[ w ][ g1 ] = new GRBVar[num_gpus];
      for ( int g2 = 0; g2 < num_gpus; g2 ++ ) {
        lambda[ w ][ g1 ][ g2 ] = model->addVar( 0.0,
                                                 1.0,
                                                 0.0,
                                                 GRB_BINARY,
                                                 "lambda_" + to_string( w ) +
                                                     "_" + to_string( g1 ) +
                                                     "_" + to_string( g2 ));
      }
    }
  }

  /* wavelength uniqueness constraints:
   * each gpu pair chooses a non-overlapping subset of waves */
  int src;
  int dst;
  for ( int w = 0; w < num_waves; w ++ ) {
    GRBLinExpr num_users = 0;
    for ( src = 0; src < num_gpus; src ++ ) {
      for ( dst = 0; dst < num_gpus; dst ++ ) {
        num_users += lambda[ w ][ src ][ dst ];
      }
    }
    model->addConstr( num_users,
                      GRB_LESS_EQUAL,
                      1.0,
                      "uniqueness_constraint_" + to_string( w ));
  }

  /* each GPU can use a wavelengths to talk/listen to
   * at most one other peer */
  for ( int w = 0; w < num_waves; w ++ ) {
    for ( src = 0; src < num_gpus; src ++ ) {
      GRBLinExpr talk = 0;
      GRBLinExpr listen = 0;
      for ( dst = 0; dst < num_gpus; dst ++ ) {
        talk += lambda[ w ][ src ][ dst ];
        listen += lambda[ w ][ dst ][ src ];
      }
      model->addConstr( talk, GRB_LESS_EQUAL, 1.0, "talk_constraint_" + to_string( w ) + "_" + to_string( src ));
      model->addConstr( listen, GRB_LESS_EQUAL, 1.0, "listen_constraint_" + to_string( w ) + "_" + to_string( src ));
    }
  }

  /* Obviously, nodes do not use precious wavelengths to
   * talk to their own self :D */
  for ( int w = 0; w < num_waves; w ++ ) {
    for ( int g = 0; g < num_gpus; g ++ ) {
      model->addConstr( lambda[ w ][ g ][ g ],
                        GRB_EQUAL,
                        0.0,
                        "politeness_constraint_" + to_string( w ) + "_" + to_string( g ));
    }
  }

  /* going to maximize min_throughput */
  GRBVar
      min_throughput = model->addVar( 0.0, GRB_INFINITY, 1.0 /* obj_func_coeff */, GRB_CONTINUOUS, "min_throughput" );
  for ( src = 0; src < num_gpus; src ++ ) {
    for ( dst = 0; dst < num_gpus; dst ++ ) {
      if ( normal_tm.get_elem( src, dst ) > 0 ) {
        GRBLinExpr total_waves_expr = 0;
        for ( int w = 0; w < num_waves; w ++ ) {
          total_waves_expr += lambda[ w ][ src ][ dst ];
        }
        model->addConstr( min_throughput,
                          GRB_LESS_EQUAL,
                          total_waves_expr / normal_tm.get_elem( src, dst ),
                          "throughput_constraint_" + to_string( src ) + to_string( dst ));
      }
    }
  }

  /* set objective */
  model->set( GRB_IntAttr_ModelSense, GRB_MAXIMIZE );
#endif // HAVE_GUROBI
  return ExitStatus::SUCCESS;
}

ExitStatus MordiaInterconnect::allocate_episode_bw( ) {
#ifdef HAVE_GUROBI
  Matrix2D< double > normal_tm( num_gpus, num_gpus );
  normalize_tm( normal_tm );
  model->reset( ); /* reset solution states */
  for ( int src = 0; src < num_gpus; src ++ ) {
    for ( int dst = 0; dst < num_gpus; dst ++ ) {
      string s = "throughput_constraint_" + to_string( src ) + to_string( dst );
      /* first, try to remove any previous constraint related to the traffic matrix */
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

      /* then, add the new traffic constraints */
      if ( normal_tm.get_elem( src, dst ) > 0 ) {
        GRBLinExpr total_waves_expr = 0;
        for ( int w = 0; w < num_waves; w ++ ) {
          total_waves_expr += model->getVarByName( "lambda_" + to_string( w ) +
              "_" + to_string( src ) +
              "_" + to_string( dst ));
        }
        model->addConstr( model->getVarByName( "min_throughput" ),
                          GRB_LESS_EQUAL,
                          total_waves_expr / normal_tm.get_elem( src, dst ),
                          "throughput_constraint_" + to_string( src ) + to_string( dst ));
      }
    }
  }
  model->update( ); /* gurobi is lazy :D */
  model->optimize( );
  for ( int src = 0; src < num_gpus; src ++ ) {
    for ( int dst = 0; dst < num_gpus; dst ++ ) {
      if ( src == 0 && dst == 0 ) episode_bw.set_elem( src, dst, 0.0 );
      for ( int w = 0; w < num_waves; w ++ ) {
        episode_bw.add_elem_by( src, dst, model->getVarByName( "lambda_" + to_string( w ) +
            "_" + to_string( src ) +
            "_" + to_string( dst )).get( GRB_DoubleAttr_X ) * double( cnfg.bwxstep_per_wave ));
      }
    }
  }
  #endif //HAVE_GUROBI

  return ExitStatus::SUCCESS;
}

ExitStatus MordiaInterconnect::offline_bw_est( std::unordered_map< Device *,
                                                                   std::unordered_map< Device *, double>> &estimate ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      estimate[ &gpus[ i ]][ &gpus[ j ]] = cnfg.num_waves * cnfg.bwxstep_per_wave / double( num_gpus );
    }
  }
  return ExitStatus::SUCCESS;
}

MordiaInterconnect::MordiaInterconnect( uint16_t dev_id,
                                        GPU *gpus,
                                        uint16_t num_gpus,
                                        TMEstimatorBase *tm_estimator,
                                        const SimConfig &cnfg,
                                        const uint16_t num_waves,
                                        const std::string log_dir ) : BaseInterconnect( dev_id,
                                                                                        gpus,
                                                                                        num_gpus,
                                                                                        tm_estimator,
                                                                                        cnfg,
                                                                                        log_dir ),
                                                                      num_waves( num_waves ) {
  setup_ilp_solver( );
}

MordiaInterconnect::~MordiaInterconnect( ) {
#ifdef HAVE_GUROBI

  for ( int w = 0; w < num_waves; w ++ ) {
    for ( int g1 = 0; g1 < num_gpus; g1 ++ ) {
      delete[] lambda[ w ][ g1 ];
    }
    delete[] lambda[ w ];
  }
  delete[] lambda;
  delete model;
  delete env;
  env = nullptr;
  model = nullptr;
  lambda = nullptr;
#endif //HAVE_GUROBI
}

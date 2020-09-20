#include "electrical_switch.hh"

ExitStatus ElectricalSwitch::find_matchings( Matrix2D< double > &tm_est, Matrix2D< double > &match ) {
  /* Parallel Iterative Matching algorithm */
  match.fill_zeros( );
  srand( time( NULL ));
  std::unordered_map< int, std::vector< int > > grant;
  std::vector< int > requests;
  int rand_index;
  for ( int dst = 0; dst < num_gpus; dst ++ ) {
    requests.clear( );
    for ( int src = 0; src < num_gpus; src ++ ) {
      if ( tm_est.get_elem( src, dst ) > 0 ) requests.push_back( src );
    }
    if ( ! requests.empty( )) {
      rand_index = rand( ) % requests.size( );
      grant[ requests[ rand_index ]].push_back( dst );
    }
  }

  /* accept one of the grants */
  int num_grants;
  for ( auto g : grant ) {
    num_grants = g.second.size( );
    rand_index = ( num_grants > 1 ? ( rand( ) % num_grants ) : 0 );
    match.set_elem( g.first, g.second[ rand_index ], 1.0 );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus ElectricalSwitch::allocate_episode_bw( ) {
  tm_estimator->update_tm_est( );
  Matrix2D< double > match( num_gpus, num_gpus );
  for ( int src = 0; src < num_gpus; src ++ ) {
    double out_traffic = 0;
    for ( int dst = src; dst < num_gpus; dst ++ ) {
      out_traffic += tm_estimator->tm_est.get_elem( src, dst );
    }
    for ( int dst = src; dst < num_gpus; dst ++ ) {
      if ( out_traffic > 0 )
        match.set_elem( src, dst, tm_estimator->tm_est.get_elem( src, dst ) / out_traffic );
    }
  }
  for ( int src = 0; src < num_gpus; src ++ ) {
    double out_traffic = 0;
    for ( int dst = src; dst < num_gpus; dst ++ ) {
      out_traffic += tm_estimator->tm_est.get_elem( dst, src );
    }
    for ( int dst = src; dst < num_gpus; dst ++ ) {
      if ( out_traffic > 0 )
        match.set_elem( dst, src, tm_estimator->tm_est.get_elem( dst, src ) / out_traffic );
    }
  }
  for ( int dst = 0; dst < num_gpus; dst ++ ) {
    for ( int src = 0; src < num_gpus; src ++ ) {
      episode_bw.set_elem( src, dst, match.get_elem( src, dst ) * bwxstep_per_port );
    }
  }
  return ExitStatus::SUCCESS;
}

//ExitStatus ElectricalSwitch::allocate_episode_bw( ) {
//  tm_estimator->update_tm_est( );
//  Matrix2D< double > match( num_gpus, num_gpus );
//  find_matchings( tm_estimator->tm_est, match ).ok( );
//  for ( int dst = 0; dst < num_gpus; dst ++ ) {
//    for ( int src = 0; src < num_gpus; src ++ ) {
////      episode_bw[ src ][ dst ] = match[ src ][ dst ] * bwxstep_per_port;
//      episode_bw.set_elem( src, dst, match.get_elem( src, dst ) * bwxstep_per_port );
//    }
//  }
//  return ExitStatus::SUCCESS;
//}

ExitStatus ElectricalSwitch::offline_bw_est( std::unordered_map< Device *,
                                                                 std::unordered_map< Device *, double>> &estimate ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      estimate[ &gpus[ i ]][ &gpus[ j ]] = bwxstep_per_port;
    }
  }
  return ExitStatus::SUCCESS;
}
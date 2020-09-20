#include "fullmesh.hh"

ExitStatus FullMeshInterconnect::offline_bw_est( std::unordered_map< Device *,
                                                                     std::unordered_map< Device *,
                                                                                         double>> &estimate ) {
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      estimate[ &gpus[ i ]][ &gpus[ j ]] = cnfg.num_waves * cnfg.bwxstep_per_wave / num_gpus;
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus FullMeshInterconnect::allocate_episode_bw( ) {
  /* Make sure it is the right time to make new bandwidth decisions */
  assert( curr_step % cnfg.dec_interval == 0 );
  for ( uint16_t i = 0; i < num_gpus; i ++ ) {
    for ( uint16_t j = 0; j < num_gpus; j ++ ) {
      episode_bw.set_elem( i, j, double( cnfg.num_waves ) * cnfg.bwxstep_per_wave / num_gpus );
    }
  }
  return ExitStatus::SUCCESS;
}



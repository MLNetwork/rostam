#include "strategy.hh"
#include "op_partitioner.hh"

ExitStatus Strategy::optimize_batchsize( CG &best_graph ) {
  double best_time = std::numeric_limits< double >::max( );
  Step best_itertime;
  uint32_t best_dp_degree, best_mp_degree, best_global_bs, best_niter;

  uint32_t dp_degree;
  uint32_t mp_degree;
  Step batch_quant_step = 30;

  auto n_gpus = interconnect->num_gpus;
  for ( auto bs_niter : batchsize_to_niter ) {
    auto global_bs = bs_niter.first;
    auto niter = bs_niter.second;
    dp_degree = 1;
    for ( int i = 0; i <= log2( n_gpus ); i ++ ) {
      if ( dp_degree > global_bs )
        break;
      uint16_t local_bs = global_bs / dp_degree;
      input_graph.set_global_batchsize( local_bs );
      mp_degree = interconnect->num_gpus / dp_degree;
      map< Op *, uint32_t > batch_max_splits;
      for ( auto e : input_graph.adj ) {
        batch_max_splits[ e.first ] = mp_degree;
      }
      OpPartitionerAttribute op_partitioner_batch( batch_quant_step, batch_max_splits, input_graph );
      CG batch_partitioned_graph;
      map< Op *, vector< Op *>> parallel_batch_ops_map;
      op_partitioner_batch.partition( batch_partitioned_graph, parallel_batch_ops_map );
      assert( input_graph.adj.size( ) == parallel_batch_ops_map.size( ));

      CG batch_param_partitioned_graph;
      batch_param_partitioned_graph = batch_partitioned_graph;

      MP pl( interconnect,
             gpus,
             interconnect->num_gpus,
             input_graph,
             parallel_batch_ops_map,
             mp_degree /* avail_gpus */,
             d_max );

      if ( pl.find_placement( batch_param_partitioned_graph ) != ExitStatus::SUCCESS ) {
        cerr << "OOM on gpus." << endl;
        batch_param_partitioned_graph.release_ops( ).ok( );
        continue;
      }
      Step est_steps;
      pl.estimate_iter_time( batch_param_partitioned_graph, est_steps );
      double total_time_minutes = double( est_steps ) * double( niter ) * cnfg.step_size_sec / 60.0;
      if ( total_time_minutes < best_time ) {
        best_time = total_time_minutes;
        best_graph = batch_param_partitioned_graph;
        best_dp_degree = dp_degree;
        best_mp_degree = mp_degree;
        best_global_bs = global_bs;
        best_itertime = est_steps;
        best_niter = niter;
      } else {
        batch_param_partitioned_graph.release_ops( ).ok( );
      }
      cout << "global_bs=" << global_bs << " "
           << "dp_degree=" << dp_degree << " "
           << "mp_degree=" << mp_degree << " "
           << "est_steps=" << est_steps << " "
           << "total_time_minutes=" << total_time_minutes << " "
           << std::endl;
      log << "global_bs=" << global_bs << " "
          << "dp_degree=" << dp_degree << " "
          << "mp_degree=" << mp_degree << " "
          << "est_steps=" << est_steps << " "
          << "total_time_minutes=" << total_time_minutes << " "
          << std::endl;
//      }
      dp_degree = dp_degree << 1;
    }
  }
  std::cout << "best_global_bs=" << best_global_bs << " "
            << "best_dp_degree=" << best_dp_degree << " "
            << "best_mp_degree=" << best_mp_degree << " "
            << "best_itertime=" << best_itertime << " "
            << "best_niter=" << best_niter << " "
            << "best_time=" << best_time << " "
            << std::endl;
  return ExitStatus::SUCCESS;
}




#include "dp.hh"

ExitStatus DP::find_placement( CG &output_graph ) {
  std::unordered_map< Op *, std::vector< Op *>> replicas;
  split_batch_dim( input_graph, output_graph, replicas );
  MemOp *mem_op;
  for ( auto op : replicas ) {
    /* place each replica on one GPU */
    for ( unsigned long i = 0; i < op.second.size( ); i ++ ) {
      op.second[ i ]->device = &gpus[ i ];
    }
  }
  /* add syncronization ops */
  for ( auto op : replicas ) {
    if ( op.first->type == OpType::MEMORY ) {
      mem_op = dynamic_cast< MemOp * >( op.first );
      if ( mem_op->mem_type == MemType::WRITEVARIABLE ) {
        add_ring_reduce( output_graph, op.second ).ok( );
      }
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus DP::num_batch_splits( Op *, uint32_t &num_splits ) {
  num_splits = dp_degree;
  return ExitStatus::SUCCESS;
}

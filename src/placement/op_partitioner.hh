#ifndef ROSTAM_SRC_PLACEMENT_OP_PARTITIONER_HH_
#define ROSTAM_SRC_PLACEMENT_OP_PARTITIONER_HH_
#include "op.hh"
#include "graph.hh"
#include "exit_status.hh"
#include "math.h"

using namespace std;

class OpPartitioner {
  Step quant_step;
  const map< Op *, uint32_t > max_splits;
 protected:
  CG input_graph;
 protected:
  ExitStatus split_compute( CompOp *op, uint32_t &num_splits );

  ExitStatus split_memory( MemOp *op, const map< Op *, uint32_t > &comp_splits_map, uint32_t &num_splits );

  ExitStatus get_nsplits_all( map< Op *, uint32_t > &splits_map );

 public:
  OpPartitioner( Step quant_step, const map< Op *, uint32_t > &max_splits, const CG &input_graph )
      : quant_step( quant_step ), max_splits( max_splits ), input_graph( input_graph ) { }

  ExitStatus partition( CG &output_graph, map< Op *, vector< Op *>> &parallel_ops_map );

  virtual ExitStatus create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                          map< Op *, vector< Op * > > &parallel_ops_map ) = 0;

  virtual ExitStatus add_data_dependencies( const map< Op *, vector< Op * > > &parallel_ops_map, CG &output_graph ) = 0;

  virtual ~OpPartitioner( ) = default;
};

class OpPartitionerSample : public OpPartitioner {
 public:
  OpPartitionerSample( Step quant_step, const map< Op *, uint32_t > &max_splits, const CG &input_graph );

  ExitStatus create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                  map< Op *, vector< Op *>> &parallel_ops_map ) override;

  ExitStatus add_data_dependencies( const map< Op *, vector< Op *>> &parallel_ops_map, CG &output_graph ) override;
};

class OpPartitionerAttribute : public OpPartitioner {
 public:
  OpPartitionerAttribute( Step quant_step, const map< Op *, uint32_t > &max_splits, const CG &input_graph );

  ExitStatus create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                  map< Op *, vector< Op *>> &parallel_ops_map ) override;

  ExitStatus add_data_dependencies( const map< Op *, vector< Op *>> &parallel_ops_map, CG &output_graph ) override;
};

class OpPartitionerParam : public OpPartitioner {
 public:
  OpPartitionerParam( Step quant_step, const map< Op *, uint32_t > &max_splits, const CG &input_graph );

  ExitStatus create_parallel_ops( const map< Op *, uint32_t > &splits_map,
                                  map< Op *, vector< Op *>> &parallel_ops_map ) override;

  ExitStatus add_data_dependencies( const map< Op *, vector< Op *>> &parallel_ops_map, CG &output_graph ) override;
};

#endif //ROSTAM_SRC_PLACEMENT_OP_PARTITIONER_HH_

#ifndef SIPML_SRC_BASE_PLACEMENT_HH_
#define SIPML_SRC_BASE_PLACEMENT_HH_
#include "op.hh"
#include "graph.hh"
#include "base_interconnect.hh"

class BasePlacement {
 protected:
  BaseInterconnect *interconnect;
  GPU *gpus;
  uint32_t num_gpus;
 protected:
  virtual ExitStatus num_batch_splits( Op *op, uint32_t &num_splits ) = 0;

  ExitStatus add_ring_reduce( CG &graph, std::vector< Op * > replicas );

  ExitStatus split_batch_dim( const CG &raw_graph,
                              CG &output_graph,
                              std::unordered_map< Op *, std::vector< Op *>> &replicas );

  ExitStatus split_param_dim( const CG &raw_graph,
                              CG &output_graph,
                              std::unordered_map< Op *, std::vector< Op *>> &replicas );

  ExitStatus add_async_netops( CG &output_graph );

//  ExitStatus add_sync_netops( CG &output_graph, std::unordered_map< Op *, std::vector< Op *>> &replicas );
  ExitStatus add_sync_netops( CG &output_graph, std::map< Op *, std::vector< Op *>> &replicas );

 public:
  virtual ExitStatus find_placement( CG &output_graph ) = 0;

  CG input_graph;

  ExitStatus estimate_iter_time( CG &graph, Step &num_steps_est );

 public:
  BasePlacement( BaseInterconnect *interconnect, GPU *gpus, uint32_t num_gpus, const CG &input_graph )
      : interconnect( interconnect ), gpus( gpus ), num_gpus( num_gpus ), input_graph( input_graph ) { }

  BasePlacement( const BasePlacement & ) = delete;

  BasePlacement &operator=( const BasePlacement & ) = delete;

  virtual ~BasePlacement( ) = default;

  ExitStatus add_dp_sync_netops( CG &output_graph, std::map< Op *, std::vector< Op *>> &replicas, int dp_degree );

  ExitStatus add_dp_ring_reduce( CG &graph, std::vector< Op * > replicas, int dp_degree );
};

#endif //SIPML_SRC_BASE_PLACEMENT_HH_

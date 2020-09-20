#ifndef ROSTAM_SRC_PLACEMENT_MP_HH_
#define ROSTAM_SRC_PLACEMENT_MP_HH_
#include "base_placement.hh"

using namespace std;

class MP : public BasePlacement {
 private:
  map< Op *, vector< Op *>> parallel_ops_map;
  int avail_gpus;
  int d_max;
 public:
  MP( BaseInterconnect *interconnect,
      GPU *gpus,
      uint32_t num_gpus,
      const CG &input_graph,
      const map< Op *, vector< Op *>> &parallel_ops_map,
      int avail_gpus,
      int d_max ) : BasePlacement( interconnect, gpus, num_gpus, input_graph ),
                    parallel_ops_map( parallel_ops_map ),
                    avail_gpus( avail_gpus ),
                    d_max( d_max ) { }

 protected:
  ExitStatus num_batch_splits( Op *op, uint32_t &num_splits ) override;

 public:
  ExitStatus find_placement( CG &output_graph ) override;

  ExitStatus place_partitions_heuristic( CG &output_graph );

  ExitStatus get_earliest_available( int &ready_dev_id,
                                     const vector< Step > &v,
                                     uint32_t start_offset,
                                     uint32_t end_offset,
                                     uint64_t mem_size );

  ExitStatus add_global_dp_ops( CG &output_graph );
};

#endif //ROSTAM_SRC_PLACEMENT_MP_HH_

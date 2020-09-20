#ifndef ROSTAM_SRC_PLACEMENT_DP_HH_
#define ROSTAM_SRC_PLACEMENT_DP_HH_
#include "base_placement.hh"

class DP : public BasePlacement {
 public:
  const uint32_t dp_degree;
 public:
  DP( BaseInterconnect *interconnect, GPU *gpus, uint32_t num_gpus, const CG &input_graph, const uint32_t dp_degree )
      : BasePlacement(
      interconnect,
      gpus,
      num_gpus,
      input_graph ), dp_degree( dp_degree ) { }

  ExitStatus find_placement( CG &output_graph ) override;

 protected:
  ExitStatus num_batch_splits( Op *, uint32_t &num_splits ) override;
};

#endif //ROSTAM_SRC_PLACEMENT_DP_HH_

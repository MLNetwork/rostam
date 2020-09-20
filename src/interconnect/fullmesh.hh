#ifndef SIPML_SRC_FULLMESH_HH_
#define SIPML_SRC_FULLMESH_HH_
#include "base_interconnect.hh"

class FullMeshInterconnect : public BaseInterconnect {
 public:
  FullMeshInterconnect( uint16_t dev_id,
                        GPU *gpus,
                        uint16_t num_gpus,
                        TMEstimatorBase *tm_estimator,
                        const SimConfig &cnfg,
                        const std::string log_dir ) :
      BaseInterconnect( dev_id,
                        gpus,
                        num_gpus,
                        tm_estimator,
                        cnfg,
                        log_dir ) { }

  ExitStatus offline_bw_est( std::unordered_map< Device *, std::unordered_map< Device *, double>> &estimate ) override;

 private:
  ExitStatus allocate_episode_bw( ) override;
};

#endif //SIPML_SRC_FULLMESH_HH_

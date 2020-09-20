#ifndef ROSTAM_SRC_INTERCONNECT_ELECTRICAL_SWITCH_HH_
#define ROSTAM_SRC_INTERCONNECT_ELECTRICAL_SWITCH_HH_
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "base_interconnect.hh"

class ElectricalSwitch : public BaseInterconnect {
  ExitStatus allocate_episode_bw( ) override;

 private:
  const double bwxstep_per_port;
 private:
  ExitStatus find_matchings( Matrix2D< double > &tm_est, Matrix2D< double > &match );

 public:
  ElectricalSwitch( uint16_t dev_id,
                    GPU *gpus,
                    uint16_t num_gpus,
                    TMEstimatorBase *tm_estimator,
                    const SimConfig &cnfg,
                    double bw_per_port,
                    const std::string log_dir ) : BaseInterconnect( dev_id,
                                                                    gpus,
                                                                    num_gpus,
                                                                    tm_estimator,
                                                                    cnfg,
                                                                    log_dir ),
                                                  bwxstep_per_port( bw_per_port * cnfg.step_size_sec )
                                                  { }

  ExitStatus offline_bw_est( std::unordered_map< Device *, std::unordered_map< Device *, double>> &estimate ) override;
};

#endif //ROSTAM_SRC_INTERCONNECT_ELECTRICAL_SWITCH_HH_

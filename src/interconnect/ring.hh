#ifndef ROSTAM_SRC_INTERCONNECT_RING_HH_
#define ROSTAM_SRC_INTERCONNECT_RING_HH_
#include "base_interconnect.hh"

using namespace std;
enum class BWDecisionType {
  ILP,
  MINCOSTFLOW
};

class RingInterconnect : public BaseInterconnect {
 private:
#ifdef HAVE_GUROBI
GRBEnv *env;
  GRBModel *model;
  GRBVar ****lambda;
#endif //HAVE_GUROBI
 public:
  const uint16_t num_waves;
  const BWDecisionType bw_decision_type;
  const int tolerable_dist;
  const int num_rings;
 public:
  RingInterconnect( uint16_t dev_id,
                    GPU *gpus,
                    uint16_t num_gpus,
                    TMEstimatorBase *tm_estimator,
                    const SimConfig &cnfg,
                    uint16_t num_waves,
                    BWDecisionType bw_decision_type,
                    const int tolerable_dist,
                    const int num_rings,
                    const std::string log_dir
  );

  RingInterconnect( const RingInterconnect & ) = delete;

  RingInterconnect &operator=( const RingInterconnect & ) = delete;

  ExitStatus offline_bw_est( std::unordered_map< Device *, std::unordered_map< Device *, double>> &estimate ) override;

  virtual ~RingInterconnect( );

 private:
  ExitStatus setup_ilp_solver( );

  ExitStatus allocate_episode_bw( ) override;

  ExitStatus allocate_episode_bw_ilp( );

  ExitStatus allocate_episode_bw_mcf( );
};

#endif //ROSTAM_SRC_INTERCONNECT_RING_HH_

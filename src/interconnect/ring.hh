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
  std::map< uint16_t, std::map< uint16_t, double > > sparse_episode_bw_budget;
  uint16_t eff_num_gpus; /* effective number of GPUs for speeding up the ILP solver */

 public:
  const uint16_t num_waves;
  const BWDecisionType bw_decision_type;
  const int tolerable_dist;
  const int num_rings;

  ExitStatus setup_ilp_solver( );

 public:
  RingInterconnect( uint16_t dev_id,
                    GPU *gpus,
                    uint16_t num_gpus,
                    double ingress_link_speed,
                    double egress_link_speed,
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

  ExitStatus is_routing_feasible( Packet* pkt, bool &is_feasible ) override;

  ExitStatus set_eff_num_gpus( uint16_t n );

  ExitStatus reset_routing_step_counters( ) override;

  virtual ~RingInterconnect( );

 private:
  ExitStatus allocate_episode_bw( ) override;

  ExitStatus allocate_episode_bw_ilp( );

  ExitStatus allocate_episode_bw_mcf( );
};

#endif //ROSTAM_SRC_INTERCONNECT_RING_HH_

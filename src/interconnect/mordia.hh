#ifndef ROSTAM_SRC_INTERCONNECT_MORDIA_HH_
#define ROSTAM_SRC_INTERCONNECT_MORDIA_HH_
#ifdef HAVE_GUROBI
#include <gurobi_c++.h>
#endif // HAVE_GUROBI
#include "base_interconnect.hh"

using namespace std;

class MordiaInterconnect : public BaseInterconnect {
 private:
#ifdef HAVE_GUROBI
GRBEnv *env;
  GRBModel *model;
  GRBVar ***lambda;
#endif //HAVE_GUROBI
 public:
  const uint16_t num_waves;
 public:
  MordiaInterconnect( uint16_t dev_id,
                      GPU *gpus,
                      uint16_t num_gpus,
                      double ingress_link_speed,
                      double egress_link_speed,
                      TMEstimatorBase *tm_estimator,
                      const SimConfig &cnfg,
                      uint16_t num_waves,
                      const std::string log_dir );

  MordiaInterconnect( const MordiaInterconnect & ) = delete;

  MordiaInterconnect &operator=( const MordiaInterconnect & ) = delete;

  ExitStatus offline_bw_est( std::unordered_map< Device *, std::unordered_map< Device *, double>> &estimate ) override;

  ExitStatus is_routing_feasible( Packet* pkt, bool &is_feasible ) override;

  virtual ~MordiaInterconnect( );

 private:
  ExitStatus setup_ilp_solver( );

  ExitStatus allocate_episode_bw( ) override;
};

#endif //ROSTAM_SRC_INTERCONNECT_MORDIA_HH_

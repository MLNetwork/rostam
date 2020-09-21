#ifndef SIPML_SRC_OCS_HH_
#define SIPML_SRC_OCS_HH_
#include "base_interconnect.hh"

using namespace std;

class OCSInterconnect : public BaseInterconnect {
 private:
#ifdef HAVE_GUROBI
GRBModel *model;
#endif
 public:
  const uint16_t num_ocs;
  const uint16_t port_count; /* #ports per ocs */
  const bool single_shot;
  map< uint16_t, map< uint16_t, Device * > > port_map; /* ocs_no -> port_no -> device */

 private:
  ExitStatus setup_optimal_solver( );

  ExitStatus setup_optimal_solver_singleshot( );

  ExitStatus setup_optimal_solver_multishot( );

  ExitStatus allocate_episode_bw( ) override;

  ExitStatus allocate_episode_bw_singleshot( );

  ExitStatus allocate_episode_bw_multishot( );

 public:
  OCSInterconnect( uint16_t dev_id,
                   GPU *gpus,
                   uint16_t num_gpus,
                   TMEstimatorBase *tm_estimator,
                   const SimConfig &cnfg,
                   const uint16_t num_ocs,
                   const uint16_t port_count,
                   const bool single_shot,
                   const std::string log_dir ) : BaseInterconnect( dev_id,
                                                                   gpus,
                                                                   num_gpus,
                                                                   tm_estimator,
                                                                   cnfg,
                                                                   log_dir ),
#ifdef HAVE_GUROBI
                                                 model( nullptr ),
#endif //HAVE_GUROBI
                                                 num_ocs( num_ocs ),
                                                 port_count( port_count ),
                                                 single_shot( single_shot ),
                                                 port_map( ) {

    /* construct the port_map */
    /* be careful about interpretting the solver solutions when this mapping changes */
    for ( int ocs_no = 0; ocs_no < num_ocs; ocs_no ++ ) {
      for ( int port_no = 0; port_no < port_count; port_no ++ ) {
        int global_port_no = ocs_no * port_count + port_no;
        int map_id = global_port_no % num_gpus; /* should correspond to the physical wirings */
        port_map[ ocs_no ][ port_no ] = &gpus[ map_id ];
      }
    }
#ifdef HAVE_GUROBI
    /* create an environment */
    GRBEnv env = GRBEnv( true );

    /* create an empty model */
    env.start( );
    model = new GRBModel( env );
    #endif //HAVE_GUROBI

    setup_optimal_solver( );
  }

  ~OCSInterconnect( ) { }

  OCSInterconnect( const OCSInterconnect & ) = delete;

  OCSInterconnect &operator=( const OCSInterconnect & ) = delete;

  ExitStatus offline_bw_est( unordered_map< Device *, unordered_map< Device *, double>> &estimate ) override;
};

#endif //SIPML_SRC_OCS_HH_

#ifndef SIPML_SRC_PLACEMENT_STRATEGY_HH_
#define SIPML_SRC_PLACEMENT_STRATEGY_HH_
#include <utility>
#include <math.h>
#include "exit_status.hh"
#include "op.hh"
#include "graph.hh"
#include "interconnect.hh"
#include "mp.hh"

class Strategy {
 private:
  const CG input_graph;
  BaseInterconnect *interconnect;
  const std::map< uint32_t, uint32_t > batchsize_to_niter;
  const Step batch_quant_step;
  GPU *gpus;
  const int d_max;
  SimConfig cnfg;
  const string log_dir;
  ofstream log;
 public:
  Strategy( const CG &input_graph,
            BaseInterconnect *interconnect,
            std::map< uint32_t, uint32_t > batchsize_to_niter,
            const Step batch_quant_step,
            GPU *gpus,
            const int d_max,
            const SimConfig cnfg,
            const string log_dir )
      : input_graph( input_graph ),
        interconnect( interconnect ),
        batchsize_to_niter( std::move( batchsize_to_niter )),
        batch_quant_step( batch_quant_step ),
        gpus( gpus ), d_max( d_max ), cnfg( cnfg ), log_dir( log_dir ), log( ) {
    log = std::ofstream( log_dir + "strategy.log", std::ofstream::out );
  }

  ExitStatus optimize_batchsize( CG &best_graph );

  ExitStatus get_hybrid_placement( uint32_t dp_degree,
                                   uint32_t mp_degree,
                                   uint32_t global_bs,
                                   Step &est_steps,
                                   CG &batch_param_partitioned_graph );

  ExitStatus raw_placement( CG &best_graph,
                            uint32_t global_bs,
                            uint32_t dp_degree,
                            Step batch_quant_step,
                            Step param_quant_step );

 private:
  ExitStatus scale_input_graph( CG &scaled_input_graph, const double &batch_factor );

 public:
  ~Strategy( ) {
    log.close( );
  }
  Strategy(const Strategy&) = delete;

  Strategy &operator=(const Strategy&) = delete;
};

#endif //SIPML_SRC_PLACEMENT_STRATEGY_HH_

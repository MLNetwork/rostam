#ifndef TEST_SIM_CONFIG_H
#define TEST_SIM_CONFIG_H
#include <assert.h>
#include <cstdint>
#include "exit_status.hh"
#include <iostream>

/* physics configuration specs */
#define BW_PER_WAVE_BYTES 25e9/8
#define MRR_RECONF_DELAY_SEC 20e-6
#define OCS_RECONF_DELAY_SEC 30e-3
#define GPU_LAUNCH_LATENCY_SEC 0
#define GPU_MIN_COMP_TIME_SEC 0
#define INTERCONNECT_LATENCY_SEC 0
#define PCIE_LATENCY_SEC 1e-6
#define GPU_MEMORY_BYTES 32e9 
using Step = uint32_t;

enum class InterType {
  RING,
  FULLMESH,
  OCS,
  ELECTSW
};

struct SimConfig {
 public:
  const uint32_t num_waves;
  const InterType inter_type;
  const uint32_t bwxstep_per_wave; /* in bytes per step */
  const Step dec_interval;
  const Step interconnect_reconf_delay;    /* in second */
  const Step gpu_launch_latency;  /* minimum steps it takes to launch a computation kernel op on the gpu */
  const Step gpu_min_comp_time;
  const Step interconnect_latency;
  const Step pcie_latency;
  const double step_size_sec; /* in second */

 public:
  SimConfig( const uint32_t num_waves,
             const InterType inter_type,
             const uint32_t bwxstep_per_wave,
             const Step dec_interval,
             const Step interconnect_reconf_delay,
             const Step gpu_launch_latency,
             const Step gpu_min_comp_time,
             const Step interconnect_latency,
             const Step pcie_latency,
             const double step_size_sec )
      : num_waves( num_waves ),
        inter_type( inter_type ),
        bwxstep_per_wave( bwxstep_per_wave ),
        dec_interval( dec_interval ),
        interconnect_reconf_delay( interconnect_reconf_delay ),
        gpu_launch_latency( gpu_launch_latency ),
        gpu_min_comp_time( gpu_min_comp_time ),
        interconnect_latency( interconnect_latency ),
        pcie_latency( pcie_latency ),
        step_size_sec( step_size_sec ) { }

  void summary( ) {
    std::cout << "num_waves=" << num_waves << " "
              << "bwxstep_per_wave=" << bwxstep_per_wave << " "
              << "dec_interval=" << dec_interval << " "
              << "interconnect_reconf_delay=" << interconnect_reconf_delay << " "
              << "gpu_launch_latency=" << gpu_launch_latency << " "
              << "gpu_min_comp_time=" << gpu_min_comp_time << " "
              << "interconnect_latency=" << interconnect_latency << " "
              << "pcie_latency=" << pcie_latency << " "
              << "step_size_sec=" << step_size_sec << std::endl;
  }
};

#endif //TEST_SIM_CONFIG_H

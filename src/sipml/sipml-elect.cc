#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <assert.h>
#include <sys/stat.h>
#include "matrix.hh"
#include "sim_config.hh"
#include "interconnect.hh"
#include "gpu.hh"
#include "session.hh"
#include "graph.hh"
#include "placement.hh"
#include "strategy.hh"
#include "utils.hh"

#define MAX_NUM_ITERATIONS 1000000

using namespace std;

uint16_t GPU::dev_count = 0;

const uint16_t Packet::max_pkt_size = 1504; /* bytes */
uint64_t Packet::num_pkts = 0;

Step Device::curr_step = 0;

NetworkSignal GPU::recv_sig = { };

std::unordered_map< PacketId, Packet * > Transport::flying_pkts = { };

static struct option command_line_options[] = {
    { "num_gpus", required_argument, nullptr, 'g' },
    { "bw_per_port_Gb", required_argument, nullptr, 'b' },
    { "latency_us", required_argument, nullptr, 'd' },
    { "strategy", required_argument, nullptr, 's' },
    { "input_profile", required_argument, nullptr, 'i' },
    { "log_dir", required_argument, nullptr, 'l' },
    { "num_profiles", required_argument, nullptr, 'n' },
    { "help", no_argument, nullptr, 'h' },
    { 0, 0, 0, 0 }
};

void usage( const char *argv0 ) {
  cerr << "Usage: " << argv0
       << " [-g,--num_gpus NUM_GPUS]"
       << " [-b,--bw_per_port_Gb BW_PER_PORT ( GIGABITS / SEC ) ]"
       << " [-d,--latency_us LATENCY_MICROSECOND ]"
       << " [-i,--input_profile INPUT_PROFILE] [-l,--log_dir LOG_DIR]" << endl
       << endl;
}

int main( int argc, char **argv ) {
  uint16_t num_gpus = 0;
  double bw_per_port_Gb = 0;
  uint32_t latency_us = 1;
  string input_profile;
  string log_dir;
  bool is_auto_strategy = true;
  uint32_t dp_degree;
  uint32_t mp_degree;
  uint32_t global_bs;
  int num_profiles = 10;
  /* parse the input options */
  while ( true ) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    const int opt = getopt_long( argc, argv, "g:b:d:s:i:l:n:h", command_line_options, &option_index );

    /* Detect the end of the options. */
    if ( opt == - 1 )
      break;
    switch ( opt ) {
      case 'g':num_gpus = stoul( optarg );
        break;
      case 'b':bw_per_port_Gb = stoul( optarg );
        break;
      case 'd':latency_us = stoul( optarg );
        break;
      case 's': {
        string strat;
        strat = optarg;
        if ( ! strat.compare( "auto" ))
          is_auto_strategy = true;
        else {
          is_auto_strategy = false;
          string delimiter = ":";
          size_t pos = strat.find( delimiter );
          dp_degree = stoul( strat.substr( 0, pos ));
          strat.erase( 0, pos + delimiter.length( ) );
          pos = strat.find( delimiter );
          mp_degree = stoul( strat.substr( 0, pos ) );
          strat.erase( 0, pos + delimiter.length( ) );
          global_bs = stoul( strat );
        }
      }
        break;
      case 'i':input_profile = optarg;
        break;
      case 'l':log_dir = optarg;
        break;
      case 'n': num_profiles = stoul( optarg );
        break;
      case 'h':usage( argv[ 0 ] );
        return EXIT_SUCCESS;
        break;
      default:usage( argv[ 0 ] );
        return EXIT_FAILURE;
    }
  }

  /* Print any remaining command line arguments (not options). */
  if ( optind < argc ) {
    printf( "non-option ARGV-elements: " );
    while ( optind < argc )
      printf( "%s ", argv[ optind ++ ] );
    putchar( '\n' );
  }

  /* create the log directory if it doesn't exists */
  string mkdir_cmnd = "mkdir -p " + log_dir;
  if ( system( mkdir_cmnd.c_str( )) == - 1 )
    cerr << "Error :  " << strerror(errno) << endl;
  else
    cout << "Directory created" << endl;
  const double step_size_sec = 1e-4;
  const uint32_t bwxstep_per_wave = BW_PER_WAVE_BYTES * step_size_sec;
  const Step gpu_launch_latency = GPU_LAUNCH_LATENCY_SEC / step_size_sec;
  const Step gpu_min_comp_time = GPU_MIN_COMP_TIME_SEC / step_size_sec;
  const Step interconnect_latency = double( latency_us ) * 1e-6 / step_size_sec;
  const Step pcie_latency = PCIE_LATENCY_SEC / step_size_sec;
  const Step dec_interval = std::numeric_limits<Step>::max();

  int num_waves = 0;
  Step interconnect_reconf_delay = 0;
  SimConfig cnfg( num_waves,
                  InterType::ELECTSW,
                  bwxstep_per_wave,
                  dec_interval,
                  interconnect_reconf_delay,
                  gpu_launch_latency,
                  gpu_min_comp_time,
                  interconnect_latency,
                  pcie_latency,
                  step_size_sec );
  cnfg.summary( );

  /* create gpus */
  auto gpus = new GPU[num_gpus];
  for ( int i = 0; i < num_gpus; i ++ ) {
    gpus[ i ].cnfg = &cnfg;
    gpus[ i ].setup_transport( num_gpus );
  }


  /* traffic matrix estimator */
  TransportEstimator tm_estimator( num_gpus, log_dir );



  /* construct an interconnect */
  double bw_per_port_bytes = bw_per_port_Gb * 1e9 / 8;
  ElectricalSwitch
      interconnect( 0 /* device_id */, gpus, num_gpus, bw_per_port_bytes, bw_per_port_bytes, &tm_estimator, cnfg, bw_per_port_bytes, log_dir );

  /* create the computation workload graph */
  CG graph;
  graph.from_graph_profile( input_profile, cnfg.step_size_sec, num_profiles );

  /* get a summary of the graph at an example batch size */
  graph.set_global_batchsize( 64 );
  graph.summary( log_dir );

  /* read the number of training iterations required
   * at each global batch size */
  std::map< uint32_t, uint32_t > bs2niter_map;
  batch2niter_map_fromfile( input_profile, bs2niter_map );

  /* create the strategy for finding the best parallelization config */
  Step batch_quant_step = 1e-6 / cnfg.step_size_sec;
  Strategy strategy( graph,
                     &interconnect,
                     bs2niter_map,
                     batch_quant_step,
                     gpus,
                     num_gpus /* max_dist */,
                     cnfg,
                     log_dir );
  CG final_graph;
  if ( is_auto_strategy )
    strategy.optimize_batchsize( final_graph ).ok( );
  else {
    Step est_steps;
    strategy.get_hybrid_placement( dp_degree, mp_degree, global_bs, est_steps, final_graph ).ok( );
    cout << "est_steps=" << est_steps << endl;
  }
  final_graph.summary( );

  /* construct the sessions */
  Session session( 0 /* session_id */, gpus, final_graph, log_dir );
  cout << "input graph size: " << graph.adj.size( ) << endl;
  tm_estimator.bind_to_transports( gpus, num_gpus );
  SingleShotEsimator single_shot_esimator( num_gpus, log_dir );
  single_shot_esimator.bind_to_sessions( &session, 1 );
  single_shot_esimator.log( );
  bool done = false;
  for ( int it = 0; it < MAX_NUM_ITERATIONS; it ++ ) {
    session.proceed( done ).ok( );
    if ( done )
      break;
    interconnect.proceed( ).ok( );
    Device::curr_step ++;
  }
  delete[] gpus;
  return 0;
}

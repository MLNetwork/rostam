#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <assert.h>
#include <sys/stat.h>
#include "sim_config.hh"
#include "interconnect.hh"
#include "gpu.hh"
#include "session.hh"
#include "graph.hh"
#include "placement.hh"
#include "strategy.hh"
#include "utils.hh"

using namespace std;

uint16_t GPU::dev_count = 0;

const uint16_t Packet::max_pkt_size = 1504; /* bytes */
uint64_t Packet::num_pkts = 0;

Step Device::curr_step = 0;

NetworkSignal GPU::recv_sig = { };

std::unordered_map< PacketId, Packet * > Transport::flying_pkts = { };

static struct option command_line_options[] = {
    { "num_gpus", required_argument, nullptr, 'g' },
    { "num_waves", required_argument, nullptr, 'w' },
    { "num_ocs", required_argument, nullptr, 'o' },
    { "port_count", required_argument, nullptr, 'p' },
    { "dec_interval_micro", required_argument, nullptr, 'm' },
    { "input_profile", required_argument, nullptr, 'i' },
    { "log_dir", required_argument, nullptr, 'l' },
    { "single_shot", no_argument, nullptr, 's' },
    { "help", no_argument, nullptr, 'h' },
    { 0, 0, 0, 0 }
};

void usage( const char *argv0 ) {
  cerr << "Usage: " << argv0
       << " [-g,--num_gpus NUM_GPUS] [-w, --num_waves NUM_WAVES]"
       << " [-p,--port_count PORT_COUNT] [-o, --num_ocs NUM_OCS]"
       << " [-i,--input_profile INPUT_PROFILE] [-l,--log_dir LOG_DIR]" << endl
       << endl;
}

int main( int argc, char **argv ) {
  uint16_t num_gpus = 0;
  uint16_t num_waves = 0;
  uint16_t num_ocs = 0;
  uint16_t port_count = 0;
  bool single_shot = false;
  double bw_dec_micro = 0;

  string input_profile;
  string log_dir;
  /* parse the input options */
  while ( true ) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    const int opt = getopt_long( argc, argv, "g:w:o:p:m:i:l:s:h", command_line_options, &option_index );

    /* Detect the end of the options. */
    if ( opt == - 1 )
      break;
    switch ( opt ) {
      case 'g':num_gpus = stoul( optarg );
        break;
      case 'w':num_waves = stoul( optarg );
        break;
      case 'o':num_ocs = stoul( optarg );
        break;
      case 'p':port_count = stoul( optarg );
      case 'm':bw_dec_micro = stoul( optarg );
      case 'i':input_profile = optarg;
        break;
      case 'l':log_dir = optarg;
        break;
      case 's':single_shot = true;
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
    cout << "Directory created." << endl;
  cout << single_shot << endl;
  const double step_size_sec = 1e-6;
  const uint32_t bwxstep_per_wave = BW_PER_WAVE_BYTES * step_size_sec;
  const Step mrr_reconf_delay = MRR_RECONF_DELAY_SEC / step_size_sec;
  const Step ocs_reconf_delay = OCS_RECONF_DELAY_SEC / step_size_sec;
  const Step gpu_launch_latency = GPU_LAUNCH_LATENCY_SEC / step_size_sec;
  const Step gpu_min_comp_time = GPU_MIN_COMP_TIME_SEC / step_size_sec;
  const Step interconnect_latency = INTERCONNECT_LATENCY_SEC / step_size_sec;
  const Step pcie_latency = PCIE_LATENCY_SEC / step_size_sec;
  Step dec_interval;
  if ( single_shot )
    dec_interval = std::numeric_limits< Step >::max( );
  else
    dec_interval = bw_dec_micro * 1e-6 / step_size_sec;
  SimConfig cnfg( num_waves,
                  InterType::OCS,
                  bwxstep_per_wave,
                  dec_interval,
                  mrr_reconf_delay,
                  ocs_reconf_delay,
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

  /* traffic matrix estimator required for configuring
   * the ocs switches at the start of the training process */
  TMEstimatorBase *tm_estimator;
  if ( single_shot )
    tm_estimator = new SingleShotEsimator( num_gpus, log_dir );
  else {
    tm_estimator = new TransportEstimator( num_gpus, log_dir );
  }
  /* construct an interconnect */
  OCSInterconnect
      interconnect( 0 /* device_id */, gpus, num_gpus, tm_estimator, cnfg, num_ocs, port_count, single_shot, log_dir );

  /* create the computation workload graph */
  CG graph;
  graph.from_graph_profile( input_profile, cnfg.step_size_sec, 10 );

  /* get a summary of the graph at an example batch size */
  graph.set_global_batchsize( 64 );
  graph.summary( );

  /* read the number of training iterations required
 * at each global batch size */
  std::map< uint32_t, uint32_t > bs2niter_map;
  batch2niter_map_fromfile( input_profile, bs2niter_map );
  uint16_t d_max = num_ocs;
  Strategy strategy( graph,
                     &interconnect,
                     bs2niter_map,
                     gpus,
                     d_max,
                     cnfg,
                     log_dir );
  CG final_graph;
  strategy.optimize_batchsize( final_graph ).ok( );

  /* construct the sessions */
  Session session( 0 /* session_id */, gpus, final_graph, log_dir );
  cout << "input graph size: " << graph.adj.size( ) << endl;
  SingleShotEsimator single_shot_esimator( num_gpus, log_dir );
  if ( single_shot ) {
    tm_estimator->bind_to_sessions( &session, 1 );
    tm_estimator->log( );
  } else {
    dynamic_cast<TransportEstimator *>(tm_estimator)->bind_to_transports( gpus, num_gpus );
    single_shot_esimator.bind_to_sessions( &session, 1 );
    single_shot_esimator.log( );
  }
  bool done = false;
  for ( int it = 0; it < 1000000; it ++ ) {
    session.proceed( done ).ok( );
    if ( done )
      break;
    interconnect.proceed( ).ok( );
    Device::curr_step ++;
  }
  delete[] gpus;
  return 0;
}

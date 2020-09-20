#ifndef SIPML_SRC_GPU_HH_
#define SIPML_SRC_GPU_HH_
#include <deque>
#include <vector>
#include <utility>
#include <map>
#include "packet.hh"
#include "device.hh"
#include "exit_status.hh"
#include "op.hh"
#include "transport.hh"

using NetworkSignal = std::unordered_map< PacketId, Op * >;

class GPU : public Device {
 private:
  std::deque< Packet * > tx_buff;
  std::deque< Packet * > rx_buff;
  Step next_available;
  /* device stats */
  const uint64_t total_memory;
  uint64_t used_memory;
  std::map< Op *, uint64_t > mem_map;
  Step load_est;
 public:
  static uint16_t dev_count;
  Transport *tp;
  /* a feature to signal when a communication Op is finished */
  static NetworkSignal recv_sig;
  SimConfig *cnfg;
 public:
  explicit GPU( ) : Device( dev_count, DeviceType::GPU ),
                    tx_buff( ),
                    rx_buff( ),
                    next_available( 0 ),
                    total_memory( GPU_MEMORY_BYTES ),
                    used_memory( 0 ),
                    mem_map( ),
                    load_est( 0 ),
                    tp( nullptr ),
                    cnfg( nullptr ){
    dev_count ++;
  }

  GPU( const GPU & ) = delete;

  GPU &operator=( const GPU & ) = delete;

  ExitStatus fetch_tx( Packet *&pkt );

  ExitStatus fill_rx( Packet *pkt, uint16_t num_pkts );

  ExitStatus compute( CompOp *op );

  ExitStatus communicate( NetOp *op );

  ExitStatus get_avail_memsize( uint64_t &size ) const;

  ExitStatus check_mem_feas( Op *op, bool &is_feas );

  ExitStatus allocate_mem( Op *op );

  ExitStatus deallocate_mem( Op *op );

  ExitStatus release_mem_all( );

  ExitStatus summary( ) const;

  ExitStatus setup_transport( int num_gpus );

 private:
  ExitStatus packetize( NetOp *op );
};

#endif //SIPML_SRC_GPU_HH_

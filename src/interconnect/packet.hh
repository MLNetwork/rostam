#ifndef TEST_PACKET_H
#define TEST_PACKET_H
#include <limits>
#include "sim_config.hh"
#include "device.hh"

using PacketId = uint64_t;

class Packet {
 public:
  const Device *src;
  const Device *dst;
  const uint16_t num_bytes;
  Step tx_time;
  Step rx_time;
  const static uint16_t max_pkt_size; /* in bytes */
  static uint64_t num_pkts;
  PacketId pkt_id;
  bool acked;
 public:
  Packet( Device *src, Device *dst, uint32_t numBytes, Step txTime ) : src( src ), dst( dst ),
                                                                       num_bytes( numBytes ),
                                                                       tx_time( txTime ),
                                                                       rx_time( std::numeric_limits< Step >::max( )),
                                                                       pkt_id( num_pkts ),
                                                                       acked( false ) {
    num_pkts ++;
  }

 private:
  /* unnecessary copies may slow down the simulator */
  Packet( const Packet & );

  Packet &operator=( const Packet & );
};

#endif //TEST_PACKET_H

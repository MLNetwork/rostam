#ifndef SIPML_SRC_TRANSPORT_HH_
#define SIPML_SRC_TRANSPORT_HH_
#include <unordered_map>
#include <deque>
#include "packet.hh"
#include "exit_status.hh"
#include "op.hh"
#include "matrix.hh"

using PacketQueue = std::deque< Packet * >;

class Transport {
 private:
  /* currently assume all packets acks are shared centeralized */
  static std::unordered_map< PacketId, Packet * > flying_pkts;
 public:
  /* couple to each device */
  PacketQueue &tx_buff;
  PacketQueue &rx_buff;
  PacketQueue tp_input;
 private:
  Matrix2D< double > tm_est;
 public:
  Transport( std::deque< Packet * > &tx_buff,
             std::deque< Packet * > &rx_buff,
             const int num_gpus )
      : tx_buff( tx_buff ), rx_buff( rx_buff ), tp_input( ), tm_est( num_gpus, num_gpus ) {
  }

  ExitStatus recv( Packet *pkt );

  ExitStatus send( PacketId &last_pkt_id );

  ExitStatus get_tm_estimate( Matrix2D< double > &tm );
};

#endif //SIPML_SRC_TRANSPORT_HH_

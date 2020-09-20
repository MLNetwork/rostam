#include <iostream>
#include "transport.hh"
#include "gpu.hh"

ExitStatus Transport::send( PacketId &last_pkt_id ) {
  while ( ! tp_input.empty( )) {
    Packet *p = tp_input.front( );
    flying_pkts.emplace( p->pkt_id, p );
    tm_est.add_elem_by( p->src->dev_id, p->dst->dev_id, p->num_bytes );
    tx_buff.push_back( p );
    last_pkt_id = p->pkt_id;
    tp_input.pop_front( );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus Transport::get_tm_estimate( Matrix2D< double > &tm ) {
  tm.copy_from( tm_est );
  return ExitStatus::SUCCESS;
}

ExitStatus Transport::recv( Packet *pkt ) {
  if ( flying_pkts.erase( pkt->pkt_id ) == 1 ) {
    pkt->acked = true;
    tm_est.sub_elem_by( pkt->src->dev_id, pkt->dst->dev_id, pkt->num_bytes );
    if ( GPU::recv_sig.count( pkt->pkt_id ) == 1 ) {
      GPU::recv_sig.at( pkt->pkt_id )->end = Device::curr_step;
      GPU::recv_sig.at( pkt->pkt_id )->status = OpStatus::FINISHED;
      GPU::recv_sig.erase( pkt->pkt_id );
    }
    /* we don't need this packet any more :D */
    delete pkt;
    return ExitStatus::SUCCESS;
  } else throw std::runtime_error( "Missing packet" );
}



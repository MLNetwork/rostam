#ifndef ROSTAM_EXECUTOR_SESSION_H
#define ROSTAM_EXECUTOR_SESSION_H
#include <fstream>
#include "op.hh"
#include "device.hh"
#include "gpu.hh"
#include "graph.hh"

class Session {
 public:
  uint16_t id;
  GPU *gpus;
  /* a directed graph that maps to successor;
   * a session's graph does not change once created */
  const CG graph;
 private:
  const std::string log_dir;
 public:
  bool has_finished_pass;
 private:
  /* maintain a set of ops that are likely to
   * change op_status at each step */
  std::set< Op * > focus_closure;
  std::ofstream log_file;
 public:
  Session( uint16_t id,
           GPU *gpus,
           CG &graph,
           const std::string log_dir )
      : id( id ), gpus( gpus ), graph( graph ), log_dir( log_dir ), has_finished_pass( true ), focus_closure( ), log_file( ) {
    restart( ).ok( ); /* initialize the graph */
    log_file = std::ofstream( log_dir + "session.log", std::ofstream::out );
  }

  Session( const Session & ) = delete;

  Session &operator=( const Session & ) = delete;

  ExitStatus proceed( bool &done );

  ExitStatus get_active_ops( std::set< Op * > &active_ops );

  ExitStatus get_upcoming_ops( std::set< Op * > &upcoming_ops );

 private:
  ExitStatus launch_op( Op *op );

  ExitStatus launch_op( CompOp *op );

  ExitStatus launch_op( NetOp *op );

  ExitStatus launch_op( CntrlOp *op );

  ExitStatus launch_op( MemOp *op );

  ExitStatus update_focus_closure( );

  ExitStatus launch_ready_ops( );

  ExitStatus restart( );

  ExitStatus log( );

  ExitStatus progress_log( );
};

#endif //ROSTAM_EXECUTOR_SESSION_H

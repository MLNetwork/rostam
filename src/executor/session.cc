#include <iostream>
#include "session.hh"

ExitStatus Session::launch_op( Op *op ) {
  switch ( op->type ) {
    case OpType::COMPUTE:launch_op( static_cast< CompOp * >( op ));
      break;
    case OpType::NETWORK:launch_op( static_cast< NetOp * >( op ));
      break;
    case OpType::MEMORY:launch_op( static_cast< MemOp * >( op ));
      break;
    case OpType::CONTROL_DEPENDENCY:launch_op( static_cast< CntrlOp * >( op ));
      break;
    default: return ExitStatus::FAILURE;
  }
  return ExitStatus::SUCCESS;
}

ExitStatus Session::launch_op( CompOp *op ) {
  assert( op->status == OpStatus::READY );
  if ( op->device->type == DeviceType::CPU ) {
    op->start = Device::curr_step;
    /* assume there are enough cpu cores available to
     * schedule each cpu op without significant waiting */
    op->end = Device::curr_step + op->comp_time;
    op->status = OpStatus::WORKING;
  } else if ( op->device->type == DeviceType::GPU ) {
    gpus[ op->device->dev_id ].compute( op );
    op->status = ( op->start == Device::curr_step ? OpStatus::WORKING : OpStatus::SCHEDULED );
    op->status = ( op->end == Device::curr_step ? OpStatus::FINISHED : op->status );
  } else return ExitStatus::FAILURE;
  return ExitStatus::SUCCESS;
}

ExitStatus Session::launch_op( NetOp *op ) {
  assert( op->session_id == id );
  assert( op->status == OpStatus::READY );
  if ( op->device->type == DeviceType::INTERCONNECT ) {
    assert( op->src_device->type == DeviceType::GPU );
    assert( op->dst_device->type == DeviceType::GPU );
    gpus[ op->src_device->dev_id ].communicate( op );
    /* the receiver GPU will set a recv signal when
     * the communication is finished and the op status
     * changes automatically */
    op->status = OpStatus::WORKING;
  } else if ( op->device->type == DeviceType::PCIE ) {
    throw std::runtime_error( "PCIE_NETWORK is not added yet." );
    //ToDo: implement the pcie
  } else return ExitStatus::FAILURE;
  return ExitStatus::SUCCESS;
}

ExitStatus Session::launch_op( CntrlOp *op ) {
  assert( op->session_id == id );
  assert( op->status == OpStatus::READY );
  op->start = Device::curr_step;
  op->end = Device::curr_step;
  op->status = OpStatus::FINISHED;
  return ExitStatus::SUCCESS;
}

ExitStatus Session::launch_op( MemOp *op ) {
  assert( op->session_id == id );
  assert( op->status == OpStatus::READY );
  op->start = Device::curr_step;
  op->end = Device::curr_step;
  op->status = OpStatus::FINISHED;
  return ExitStatus::SUCCESS;
}

ExitStatus Session::update_focus_closure( ) {
  /* First, update the finished ops network ops
   * status is handled via the gpu signal. We only
   * need to handle the compute ops which are previously
   * scheduled to finish at this step. */
  std::set< Op * > ops_to_remove;
  for ( auto op : focus_closure ) {
    /* make sure all finished ops are in
     * correct status */
    if ( op->end <= Device::curr_step ) {
      op->status = OpStatus::FINISHED;
    }
    /* expand the closure by adding successors
     * to the finished ops */
    if ( op->status == OpStatus::FINISHED ) {
      for ( auto succ : graph.adj.at( op )) {
        focus_closure.emplace( succ );
      }
      /* remove the finished op from the closure */
      ops_to_remove.emplace( op );
    }
  }
  for ( auto op : ops_to_remove ) {
    focus_closure.erase( op );
  }
  return ExitStatus::SUCCESS;
}

ExitStatus Session::launch_ready_ops( ) {
//  std::string suspect ("resnet_v1_50/logits/BiasAdd_0");
  for ( auto op : focus_closure ) {
    assert( op->session_id == id );
    if ( op->status == OpStatus::WAITING ) {
      uint16_t i = 0;
      for ( auto pred : graph.reverse_adj.at( op )) {
        if ( pred->status != OpStatus::FINISHED )
          break;
        i ++;
      }
      if ( i == graph.reverse_adj.at( op ).size( )) {
        op->status = OpStatus::READY;
        launch_op( op ).ok( );
      }
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus Session::restart( ) {
  has_finished_pass = false;
  for ( auto node : graph.adj ) {
    node.first->start = std::numeric_limits< Step >::max( );
    node.first->end = std::numeric_limits< Step >::max( );
    node.first->status = OpStatus::WAITING;
    /* add the roots */
    if ( graph.reverse_adj.at( node.first ).empty( )) {
      focus_closure.emplace( node.first );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus Session::proceed( bool &done ) {
  if ( has_finished_pass )
    restart( ).ok( );

  //ToDo: make sure control_dependency ops are not wasting too many simulation steps
  update_focus_closure( ).ok( );
  launch_ready_ops( ).ok( );

  /* when an iteration is finished */
  if ( focus_closure.empty( )) {
    std::cout << "Finished a pass at step=" << Device::curr_step
              << std::endl;
    has_finished_pass = true;
    done = true;
    log( );
  }
  if ( Device::curr_step % 1000 == 0 )
    progress_log( ).ok( );
  return ExitStatus::SUCCESS;
}

ExitStatus Session::log( ) {
  log_file << Device::curr_step << std::endl;
  log_file.close( );
  return ExitStatus::SUCCESS;
}

ExitStatus Session::progress_log( ) {
  int num_finished_ops = 0;
  for ( auto e : graph.adj ) {
    if ( e.first->status == OpStatus::FINISHED )
      num_finished_ops ++;
  }
  std::cout << "[ session ]" << " "
            << "curr_step=" << Device::curr_step << " "
            << "num_finished_ops=" << num_finished_ops << " "
            << "total_ops=" << graph.adj.size( ) << " "
            << std::endl;
  return ExitStatus::SUCCESS;
}

ExitStatus Session::get_active_ops( std::set< Op * > &active_ops ) {
  for ( auto op : focus_closure ) {
    if ( op->status == OpStatus::WORKING ) {
      active_ops.emplace( op );
    }
  }
  return ExitStatus::SUCCESS;
}

ExitStatus Session::get_upcoming_ops( std::set< Op * > &upcoming_ops ) {
  for ( auto op : focus_closure ) {
    if ( op->status == OpStatus::READY ||
        op->status == OpStatus::WAITING ||
        op->status == OpStatus::SCHEDULED ) {
      upcoming_ops.emplace( op );
    }
  }
  return ExitStatus::SUCCESS;
}

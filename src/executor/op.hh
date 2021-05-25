#ifndef TEST_OP_H
#define TEST_OP_H
#include <limits>
#include <set>
#include <unordered_map>
#include <map>
#include <string>
#include <utility>
#include "device.hh"
#include "sim_config.hh"

enum class OpType {
  COMPUTE,
  MEMORY,
  NETWORK,
  CONTROL_DEPENDENCY
};
enum class OpStatus {
  WAITING,    /* waiting to meet the input dependencies */
  READY,      /* input dependencies are met */
  SCHEDULED,  /* op is scheduled to launch at the earliest availability */
  WORKING,    /* op is launched, but not finished yet */
  FINISHED    /* finished, and the output is ready to get used by the op dependents */
};

class Op {
 public:
  const std::string name;
  OpType type;
  /* which device does this op belong to */
  Device *device = device;
  Step start;
  Step end;
  OpStatus status;
  uint16_t session_id;
  const std::string creator;
  uint32_t priority;
 public:
  Op( std::string name,
      OpType type,
      Device *device,
      uint16_t sessionId,
      std::string creator )
      : name( std::move( name )),
        type( type ),
        device( device ),
        start( ),
        end( ),
        status( OpStatus::WAITING ),
        session_id( sessionId ),
        creator( std::move( creator ) ),
        priority( 0 ) { }

//  ExitStatus duplicate( uint32_t n_copies, std::vector< Op* > &new_ops );
  virtual ~Op( ) = default;


  Op( const Op & ) = default;

  Op &operator=( const Op & ) = delete;

 public:
  virtual ExitStatus get_mem_size( uint64_t &size ) const = 0;
};

class CompOp : public Op {
 public:
  std::map< uint16_t, Step > comp_time_map;
  std::map< uint16_t, uint32_t > output_bytes_map;
  Step comp_time;
  uint32_t output_bytes;

 private:
  uint16_t batch_size;

 public:
  CompOp( std::string name,
          OpType type,
          Device *device,
          uint16_t session_id,
          std::string creator,
          std::map< uint16_t, Step > comp_time_map,
          std::map< uint16_t, uint32_t > output_bytes_map )
      : Op( std::move( name ), type, device, session_id, creator ),
        comp_time_map( comp_time_map ),
        output_bytes_map( output_bytes_map ),
        comp_time( 0 ),
        output_bytes( 0 ),
        batch_size ( 0 ) { }

  CompOp( ) : Op( "", OpType::COMPUTE, nullptr, 0, "" ), comp_time_map( ), output_bytes_map( ), comp_time( 0 ), output_bytes( 0 ), batch_size( 0 ) { }

  void copy_scale_to( CompOp *new_op, double batch_scale ) const;

  ExitStatus get_mem_size( uint64_t &size ) const override;

  ExitStatus set_batch_size( uint16_t batch_size );

  ExitStatus get_batch_size( uint16_t &bs ) const;

  ExitStatus add_comp_time( uint16_t bs, Step comp_time_steps );

  ExitStatus add_output_bytes( uint16_t bs, uint32_t bytes );

};

class NetOp : public Op {
 public:
  double transfer_bytes;
  /* for network ops */
  Device *src_device = src_device;
  Device *dst_device = dst_device;
 public:
  NetOp( const std::string &name,
         OpType type,
         Device *device,
         uint16_t session_id,
         double transfer_bytes,
         Device *src_device,
         Device *dst_device,
         std::string creator ) : Op( name, type, device, session_id, creator ),
                                 transfer_bytes( transfer_bytes ),
                                 src_device( src_device ),
                                 dst_device( dst_device ) { }

  NetOp( const NetOp & ) = default;

  NetOp &operator=( const NetOp & ) = delete;

  ExitStatus get_mem_size( uint64_t &size ) const override;
};

class CntrlOp : public Op {
 public:
  CntrlOp( const std::string &name, OpType type, Device *device, uint16_t session_id, std::string creator ) : Op( name,
                                                                                                                  type,
                                                                                                                  device,
                                                                                                                  session_id,
                                                                                                                  creator ) { }

  ExitStatus get_mem_size( uint64_t &size ) const override;
};

enum class MemType {
  WRITEVARIABLE, /* to store trainable model parameters */
  READVARIABLE,
  INVALID
};

class MemOp : public Op {
 public:
  MemOp( const std::string &name,
         OpType type,
         Device *device,
         uint16_t session_id,
         MemType mem_type,
         uint32_t num_bytes,
         std::string creator ) : Op( name, type, device, session_id, creator ),
                                 mem_type( mem_type ),
                                 num_bytes( num_bytes ) { }

 public:
  MemType mem_type;
  uint32_t num_bytes;
 public:
  ExitStatus get_mem_size( uint64_t &size ) const override;
};

#endif //TEST_OP_H

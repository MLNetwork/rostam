#ifndef TEST_DEVICE_H
#define TEST_DEVICE_H
#include "sim_config.hh"

enum class DeviceType {
  GPU,
  CPU,
  PCIE,
  INTERCONNECT,
};

class Device {
 public:
  Device( uint16_t dev_id, DeviceType type ) : dev_id( dev_id ), type( type ) { }

  virtual ~Device( ) { }

 public:
  uint16_t dev_id;
  DeviceType type;
  static Step curr_step;
};

#endif

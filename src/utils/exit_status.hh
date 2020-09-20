#ifndef TEST_EXIT_STATUS_H
#define TEST_EXIT_STATUS_H
#include <cstdint>
#include <stdexcept>
#include <assert.h>

class ExitStatus {
 public:
  enum Value : uint8_t {
    SUCCESS,
    NOT_AVAILABLE,
    FAILURE
  };

  ExitStatus( ) = default;

  constexpr ExitStatus( Value status ) : value( status ) { }

  bool operator==( const ExitStatus &rhs ) const {
    return value == rhs.value;
  }

  bool operator!=( const ExitStatus &rhs ) const {
    return ! ( rhs == *this );
  }

  void ok( ) const {
    if ( value != SUCCESS ) {
      throw std::runtime_error( "Function did not run successfully." );
    }
  }

 private:
  Value value;
};

#endif //TEST_EXIT_STATUS_H

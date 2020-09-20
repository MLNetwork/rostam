#ifndef ROSTAM_SRC_UTILS_UTILS_HH_
#define ROSTAM_SRC_UTILS_UTILS_HH_
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

void batch2niter_map_fromfile( const std::string input_profile, map<uint32_t, uint32_t> &batch2niter_map ) {
  std::fstream file( input_profile + "_iter.prof", std::ios_base::in );
  if ( ! file.is_open( )) {
    throw runtime_error( "couldn't read the iteration profiles." );
  }
  uint32_t bs;
  uint32_t niter;
  string line;
  while ( getline( file, line )) {
    istringstream ls( line );
    ls >> bs >> niter;
    batch2niter_map[ bs ] = niter;
  }
  file.close( );
}

#endif //ROSTAM_SRC_UTILS_UTILS_HH_

#ifndef SIPML_SRC_GRAPH_HH_
#define SIPML_SRC_GRAPH_HH_
#include <iostream>
#include <fstream>
#include <list>
#include <stack>
#include <algorithm>
#include "exit_status.hh"
#include "op.hh"
#include "graph_profile.pb.h"

template< class NodeType >
class Graph {
 private:
  std::map< NodeType, bool > visited;
 public:
  std::map< NodeType, std::set< NodeType > > adj;
  /* a map to predecessors ( memory redundancy :D )*/
  std::map< NodeType, std::set< NodeType > > reverse_adj;

  std::vector< NodeType > sorted;
 private:
  ExitStatus dfs( NodeType u, std::stack< NodeType > &stack );

  ExitStatus find_descendants( NodeType u, std::unordered_map< NodeType, std::unordered_set< NodeType>> &descendants );

 public:
  Graph( ) : visited( ), adj( ), reverse_adj( ), sorted( ) { }

  ExitStatus add_edge( NodeType u, NodeType v );

  ExitStatus topological_sort( std::stack< NodeType > &stack );

  virtual ExitStatus summary( ) const;

  virtual ~Graph( ) = default;

  ExitStatus descendants_map( std::unordered_map< NodeType, std::unordered_set< NodeType>> &descendants );
};

template< class NodeType >
ExitStatus Graph< NodeType >::add_edge( NodeType u, NodeType v ) {
  adj[ v ]; /* create dst node if it already doesn't exist */
  adj[ u ].emplace( v );

  /* also create the reverse access graph for dependency check
   * speed-ups */
  reverse_adj[ u ];
  reverse_adj[ v ].emplace( u );
  return ExitStatus::SUCCESS;
}

template< class NodeType >
ExitStatus Graph< NodeType >::dfs( NodeType u, std::stack< NodeType > &stack ) {
  visited.at( u ) = true;
  for ( auto v : adj[ u ] ) {
    if ( ! visited.at( v ))
      dfs( v, stack );
  }
  stack.push( u );
  return ExitStatus::SUCCESS;
}

template< class NodeType >
ExitStatus Graph< NodeType >::topological_sort( std::stack< NodeType > &stack ) {
  visited.clear( );
  for ( auto i : adj )
    visited.emplace( i.first, false );
  for ( auto v : visited )
    if ( ! v.second )
      dfs( v.first, stack ).ok( );
  return ExitStatus::SUCCESS;
}

template< class NodeType >
ExitStatus Graph< NodeType >::find_descendants( NodeType u,
                                                std::unordered_map< NodeType,
                                                                    std::unordered_set< NodeType > > &descendants ) {
  if ( descendants.count( u ) == 0 ) {
    descendants[ u ];
    for ( auto v : adj.at( u )) {
      find_descendants( v, descendants ).ok( );
      descendants[ u ].emplace( v );
      descendants[ u ].insert( descendants.at( v ).begin( ), descendants.at( v ).end( ));
    }
  }
  return ExitStatus::SUCCESS;
}

template< class NodeType >
ExitStatus Graph< NodeType >::descendants_map( std::unordered_map< NodeType,
                                                                   std::unordered_set< NodeType > > &descendants ) {
  for ( auto e : adj ) {
    if ( descendants.count( e.first ) == 0 )
      find_descendants( e.first, descendants ).ok( );
  }
  return ExitStatus::SUCCESS;
}

template< class NodeType >
ExitStatus Graph< NodeType >::summary( ) const {
  for ( auto n : adj ) {
    std::cout << n.first << ": ";
    for ( auto s : n.second ) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
  }
  return ExitStatus::SUCCESS;
}

class CG : public Graph< Op * > {
 public:
  ExitStatus summary( ) const override;

  ExitStatus release_ops( );

  virtual ~CG( ) = default;

  ExitStatus scale_graph( CG &scaled_graph, const double &batch_factor ) const;

  ExitStatus priority_sort( std::map< uint32_t, Op * > &prior_sorted );

  ExitStatus critical_path_len( int &max_depth );

  ExitStatus critical_path_load( Step &load );

  ExitStatus set_global_batchsize( uint16_t bs ) const;

  ExitStatus from_graph_profile( std::string filename, const double step_size_sec, const int num_profiles );

  ExitStatus summary( std::string log_dir );
};

#endif //SIPML_SRC_GRAPH_HH_

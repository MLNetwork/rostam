#ifndef ROSTAM_SRC_UTILS_MATRIX_HH_
#define ROSTAM_SRC_UTILS_MATRIX_HH_
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <iostream>

template< class T >
class Matrix2D;

template< class T >
std::ostream &operator<<( std::ostream &, const Matrix2D< T > & );

template< class dtype >
class Matrix2D {
 public:
  Matrix2D( size_t n_cols, size_t n_rows ) : n_cols( n_cols ), n_rows( n_rows ), mat( nullptr ) {
    mat = new dtype *[n_rows];
    for ( size_t row = 0; row < n_rows; row ++ )
      mat[ row ] = new dtype[n_cols];
    fill_zeros( );
  }

  virtual ~Matrix2D( ) {
    for ( size_t row = 0; row < n_rows; row ++ )
      delete[] mat[ row ];
    delete[] mat;
  }

  Matrix2D( const Matrix2D & ) = delete;
  Matrix2D &operator=( const Matrix2D & ) = delete;

 private:
  size_t n_cols;
  size_t n_rows;
  dtype **mat;
 public:
  void fill_zeros( );

  dtype get_elem( size_t row, size_t col ) const;

  void set_elem( size_t row, size_t col, const dtype value );

  void add_by( const Matrix2D< dtype > &matrix_2_d );

  void sub_by( const Matrix2D< dtype > &matrix_2_d );

  void mul_by( const dtype &value );

  void add_elem_by( size_t row, size_t col, const dtype value );

  void sub_elem_by( size_t row, size_t col, const dtype value );

  void copy_from( const Matrix2D< dtype > &matrix_2_d );

  void normalize_by_max( );

  friend std::ostream &operator
  <<< dtype >(
  std::ostream &os,
  const Matrix2D< dtype > &d
  );
};

template< class dtype >
void Matrix2D< dtype >::fill_zeros( ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] = 0;
  }
}

template< class dtype >
void Matrix2D< dtype >::add_by( const Matrix2D< dtype > &matrix_2_d ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ ) {
      mat[ row ][ col ] += matrix_2_d.get_elem( row, col );
    }
  }
}

template< class dtype >
void Matrix2D< dtype >::sub_by( const Matrix2D< dtype > &matrix_2_d ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] -= matrix_2_d.get_elem( row, col );
  }
}

template< class dtype >
void Matrix2D< dtype >::mul_by( const dtype &value ) {
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] *= value;
  }
}

template< class dtype >
dtype Matrix2D< dtype >::get_elem( size_t row, size_t col ) const {
  return mat[ row ][ col ];
}

template< class dtype >
void Matrix2D< dtype >::copy_from( const Matrix2D< dtype > &matrix_2_d ) {
  if ( matrix_2_d.n_cols != n_cols || matrix_2_d.n_rows != n_rows ) {
    throw std::runtime_error( "matrix dimensions do not match for copying." );
  }
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      mat[ row ][ col ] = matrix_2_d.get_elem( row, col ); //todo:use memcpy
  }
}

template< class dtype >
void Matrix2D< dtype >::normalize_by_max( ) {
  dtype max_data = std::numeric_limits< dtype >::min( );
  for ( size_t row = 0; row < n_rows; row ++ ) {
    for ( size_t col = 0; col < n_cols; col ++ )
      max_data = ( max_data < mat[ row ][ col ] ? mat[ row ][ col ] : max_data );
  }
  if ( max_data != 0 ) {
    for ( size_t row = 0; row < n_rows; row ++ ) {
      for ( size_t col = 0; col < n_cols; col ++ )
        mat[ row ][ col ] /= max_data;
    }
  }
}

template< class dtype >
void Matrix2D< dtype >::set_elem( size_t row, size_t col, const dtype value ) {
  mat[ row ][ col ] = value;
}

template< class dtype >
void Matrix2D< dtype >::add_elem_by( size_t row, size_t col, const dtype value ) {
  mat[ row ][ col ] += value;
}

template< class dtype >
void Matrix2D< dtype >::sub_elem_by( size_t row, size_t col, const dtype value ) {
  mat[ row ][ col ] -= value;
}

template< class dtype >
std::ostream &operator<<( std::ostream &os, const Matrix2D< dtype > &d ) {
  for ( size_t row = 0; row < d.n_rows; row ++ ) {
    for ( size_t col = 0; col < d.n_cols; col ++ )
      os << d.mat[ row ][ col ] << " ";
    os << std::endl;
  }
  return os;
}

#endif //ROSTAM_SRC_UTILS_MATRIX_HH_

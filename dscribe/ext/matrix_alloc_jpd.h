/// @file    matrix_alloc_jpd.hpp
/// @brief   Short functions for allocating 2-dimensional C-style ** arrays.
///          Perhaps it is useful to put this short code in a separate file 
///          to make it independently of "jacobi_pd.hpp".
/// @author  Andrew Jewett
/// @license CC0-1.0

#include<cassert>

#ifndef _MATRIX_ALLOC_JPD_H
#define _MATRIX_ALLOC_JPD_H

namespace matrix_alloc_jpd {

/// @brief  Allocate a 2-dimensional array.  (Uses row-major order.)
template<typename Entry>
void Alloc2D(size_t nrows,          //!< size of the array (number of rows)
             size_t ncols,          //!< size of the array (number of columns)
             Entry ***paaX          //!< pointer to a 2D C-style array
             );

/// @brief  Deallocate arrays that were created using Alloc2D().
template<typename Entry>
void Dealloc2D(Entry ***paaX        //!< pointer to a 2D C-style array
               );



// ---- IMPLEMENTATION ----


template<typename Entry>
void Alloc2D(size_t nrows,          // size of the array (number of rows)
             size_t ncols,          // size of the array (number of columns)
             Entry ***paaX)         // pointer to a 2D C-style array
{
  assert(paaX);
  *paaX = new Entry* [nrows];  //conventional 2D C array (pointer-to-pointer)
  (*paaX)[0] = new Entry [nrows * ncols];  // 1D C array (contiguous memory)
  for(size_t iy=0; iy<nrows; iy++)
    (*paaX)[iy] = (*paaX)[0] + iy*ncols;
  // The caller can access the contents using (*paaX)[i][j]
}

template<typename Entry>
void Dealloc2D(Entry ***paaX)       // pointer to a 2D C-style array
{
  if (paaX && *paaX) {
    delete [] (*paaX)[0];
    delete [] (*paaX);
    *paaX = nullptr;
  }
}

} // namespace matrix_alloc_jpd

#endif //#ifndef _MATRIX_ALLOC_JPD_H

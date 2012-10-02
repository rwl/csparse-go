// Copyright (C) 2006, Timothy A. Davis.
// Copyright (C) 2012, Richard W. Lincoln.
//
// CSparse is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation;
// either version 2.1 of the License, or (at your option) any
// later version.
//
// CSparse is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General
// Public License along with this Module; if not, write to the
// Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
// Boston, MA 02110-1301 USA

// CSparse: a Concise Sparse matrix package.
// http://www.cise.ufl.edu/research/sparse/CSparse
package csparse

import (
)

// Matrix in compressed-column or triplet form.
type Matrix {
    // maximum number of entries
    Nzmax int
    // number of rows
    M int
    // number of columns
    N int
    // column pointers (size n+1) or col indices (size nzmax)
    P []int
    // row indices, size nzmax
    I []int
    // numerical values, size nzmax
    X []float64
    // # of entries in triplet matrix, -1 for compressed-col
    Nz int
}

// Output of symbolic Cholesky, LU, or QR analysis.
type Symbolic {
    // inverse row perm. for QR, fill red. perm for Chol
    Pinv []int
    // fill-reducing column permutation for LU and QR
    Q [] int
    // elimination tree for Cholesky and QR
    Parent []int
    // column pointers for Cholesky, row counts for QR
    Cp []int
    // leftmost[i] = min(find(A(i,:))), for QR
    Leftmost []int
    // # of rows for QR, after adding fictitious rows
    M2 int
    // # entries in L for LU or Cholesky; in V for
    Lnz int
    // # entries in U for LU; in R for QR
    Unz int
}

// Output of numeric Cholesky, LU, or QR factorization
type Numeric {
    // L for LU and Cholesky, V for QR
    L Matrix
    // U for LU, R for QR, not used for Cholesky
    U Matrix
    // partial pivoting for LU
    Pinv []int
    // beta [0..n-1] for QR
    B []float64
}

// Output of Dulmage-Mendelsohn decomposition
type Decomposition {
    // size m, row permutation
    P []int
    // size n, column permutation
    Q []int
    // size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q)
    R []int
    // size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q)
    S []int
    // # of blocks in fine dmperm decomposition
    Nb int
    // coarse row decomposition
    Rr []int
    // coarse column decomposition
    Cc []int
}

// Add sparse matrices.
// Returns C = alpha*A + beta*B, nil on error.
func Add(A, B Matrix, alpha, beta float64) Matrix {
    p, j, nz = 0, anz, m, n, bnz int
    values bool
    Cp, Ci, Bp, w []int
    x, Bx, Cx float64
    C Matrix
    if !Csc(A) || !Csc(B) {
        return nil
    }
    if A.M != B.M || A.N != B.N {
        return nil
    }
    m = A.M, anz = A.P[A.N]
    n = B.N, Bp = B.P, Bx = B.X, bnz = Bp[n]
    // get workspace
    w = make([]int, m)
    values = A.X != nil && Bx != nil
    // get workspace
    if values {
        x = make([]float64, m)
    } else {
        x = nil
    }
    // allocate result
    C = Spalloc(m, n, anz + bnz, values, 0)
    if !C || !w || (values && !x) {
        return Done(C, w, x, 0)
    }
    Cp = C.P Ci = C.I Cx = C.X
    for j := 0; j < n; j++ {
        // column j of C starts here
        Cp[j] = nz
        // alpha*A(:,j)
        nz = Scatter(A, j, alpha, w, x, j+1, C, nz)
        // beta*B(:,j)
        nz = Scatter(B, j, beta, w, x, j+1, C, nz)
        if values {
            for p := Cp[j]; p < nz; p++ {
                Cx[p] = x[Ci[p]]
            }
        }
    }
    // finalize the last column of C
    Cp[n] = nz
    return Done(C, w, x, 1)
}

import numpy as np
import scipy.sparse.linalg as la_sparse
def dg1d_poisson ( nel, ss, penal, f ):
  locdim = 3

  Amat = nel * np.array ( [ \
    [ 0.0, 0.0, 0.0 ], \
    [ 0.0, 4.0, 0.0 ], \
    [ 0.0, 0.0, 16.0 / 3.0 ]])

  Bmat = nel * np.array ( [ \
    [            penal,            1.0 - penal,            - 2.0 + penal ], \
    [     - ss - penal,     - 1.0 + ss + penal,         2.0 - ss - penal ], \
    [ 2.0 * ss + penal, 1.0 - 2.0 * ss - penal, - 2.0 + 2.0 * ss + penal ] ])

  Cmat = nel * np.array ( [ \
    [            penal,            - 1.0 + penal,            - 2.0 + penal ], \
    [       ss + penal,       - 1.0 + ss + penal,       - 2.0 + ss + penal ], \
    [ 2.0 * ss + penal, - 1.0 + 2.0 * ss + penal, - 2.0 + 2.0 * ss + penal ] ])

  Dmat = nel * np.array ( [ \
    [            - penal,            - 1.0 + penal,            2.0 - penal ], \
    [       - ss - penal,       - 1.0 + ss + penal,       2.0 - ss - penal ], \
    [ - 2.0 * ss - penal, - 1.0 + 2.0 * ss + penal, 2.0 - 2.0 * ss - penal ] ])

  Emat = nel * np.array ( [ \
    [            - penal,            1.0 - penal,            2.0 - penal ], \
    [         ss + penal,     - 1.0 + ss + penal,     - 2.0 + ss + penal ], \
    [ - 2.0 * ss - penal, 1.0 - 2.0 * ss - penal, 2.0 - 2.0 * ss - penal ] ])

  F0mat = nel * np.array ( [ \
    [              penal,              2.0 - penal,            - 4.0 + penal ], \
    [ - 2.0 * ss - penal, - 2.0 + 2.0 * ss + penal,   4.0 - 2.0 * ss - penal ], \
    [   4.0 * ss + penal,   2.0 - 4.0 * ss - penal, - 4.0 + 4.0 * ss + penal ] ])

  FNmat = nel * np.array ( [ \
    [            penal,            - 2.0 + penal,            - 4.0 + penal ], \
    [ 2.0 * ss + penal, - 2.0 + 2.0 * ss + penal, - 4.0 + 2.0 * ss + penal ], \
    [ 4.0 * ss + penal, - 2.0 + 4.0 * ss + penal, - 4.0 + 4.0 * ss + penal ] ])

  glodim = nel * locdim
  A = np.zeros ( ( glodim, glodim ) )
  b = np.zeros ( glodim )

  ng = 2
  wg = np.array ( [ 1.0, 1.0 ] )
  sg = np.array ( [ -0.577350269189, 0.577350269189 ] )
#
#  Assemble global matrix and RHS.
#  First subinterval.
#
  for ii in range ( 0, locdim ):
    for jj in range ( 0, locdim ):
      A[ii][jj] = A[ii][jj] + Amat[ii][jj] + F0mat[ii][jj] + Cmat[ii][jj]
      je = locdim + jj
      A[ii][je] = A[ii][je] + Dmat[ii][jj]

  b[0] = nel * penal
  b[1] = nel * penal * ( -1.0 ) - ss * 2.0 * nel
  b[2] = nel * penal + ss * 4.0 * nel

  for ig in range ( 0, ng ):
    xval = ( sg[ig] + 1.0 ) / ( 2.0 * nel )
    b[0] = b[0] + wg[ig] * f ( xval ) / ( 2.0 * nel ) * 1.0
    b[1] = b[1] + wg[ig] * f ( xval ) / ( 2.0 * nel ) * sg[ig]
    b[2] = b[2] + wg[ig] * f ( xval ) / ( 2.0 * nel ) * sg[ig] * sg[ig]
#
#  Intermediate subintervals.
#
  for i in range ( 2, nel ):
    for ii in range ( 0, locdim ):
      ie = ii + ( i - 1 ) * locdim
      for jj in range ( 0, locdim ):
        je = jj + ( i - 1 ) * locdim
        A[ie][je] = A[ie][je] + Amat[ii][jj] + Bmat[ii][jj] + Cmat[ii][jj]
        je = jj + ( i - 2 ) * locdim
        A[ie][je] = A[ie][je] + Emat[ii][jj]
        je = jj + i * locdim
        A[ie][je] = A[ie][je] + Dmat[ii][jj]

      for ig in range ( 0, ng ):
        xval = ( sg[ig] + 2.0 * ( i - 1 ) + 1.0 ) / ( 2.0 * nel )
        b[ie] = b[ie] + wg[ig] * f ( xval ) / ( 2.0 * nel ) * ( sg[ig] ** ii ) 
#
#  Last subinterval.
#
  for ii in range ( 0, locdim ):
    ie = ii + ( nel - 1 ) * locdim
    for jj in range ( 0, locdim ):
      je = jj + ( nel - 1 )* locdim
      A[ie][je] = A[ie][je] + Amat[ii][jj] + FNmat[ii][jj] + Bmat[ii][jj]
      je = jj + ( nel - 2 ) * locdim
      A[ie][je] = A[ie][je] + Emat[ii][jj]

    for ig in range ( 0, ng ):
      xval = ( sg[ig] + 2.0 * ( nel - 1 ) + 1.0 ) / ( 2.0 * nel )
      b[ie] = b[ie] + wg[ig]* f ( xval ) / ( 2.0 * nel ) * ( sg[ig] ** ii ) 
#
#  Solve the linear system.
# 
  c = np.linalg.solve ( A, b )
  return c

def dg1d_poisson_monomial ( x, i, nel, order ):
  h = 1.0 / nel
  xl = i * h
  xr = ( i + 1 ) * h
  xm = 0.5 * ( xl + xr )
  value = ( 2.0 * ( x - xm ) / h ) ** order
  return value

def dg1d_poisson_interp ( x, i, nel, order, c ):
  '''here order == 2'''
  value = 0.0
  for k in range ( 0, order ):
    value = value + c[k+i*order] * dg1d_poisson_monomial ( x, i, nel, k )

  return value



def dg1d_poisson_test ( ):
  import matplotlib.pyplot as plt
  import numpy as np
  import platform

  print ( '' )
  print ( 'DG1D_POISSON_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  DG1D_POISSON solves a 1D Poisson problem using the' )
  print ( '  discontinuous Galerkin method.' )

  nel = 5
  h = 1.0 / nel
  ss = 1
  penal = 5
  locdim = 3
#
#  Report parameters.
#
  print ( '' )
  print ( '  0.0 < x < 1.0' )
  print ( '  Number of subintervals = %d' % ( nel ) )
  print ( '  Number of monomials in expansion = %d' % ( locdim ) )
  print ( '  Penalty parameter = %g' % ( penal ) )
  print ( '  DG choice = %g' % ( ss ) )
  c = dg1d_poisson ( nel, ss, penal, dg1d_poisson_test_source )

  m = 10

  xh = np.zeros ( m * nel )
  uh = np.zeros ( m * nel )

  order = locdim
  k = 0
  for i in range ( 0, nel ):
    xl = float ( i       ) / float ( nel )
    xr = float ( i + 1   ) / float ( nel )
    for j in range ( 0, m ):
      xh[k] = ( ( m - 1 - j ) * xl + j * xr ) / float ( m - 1 )
      uh[k] = dg1d_poisson_interp ( xh[k], i, nel, order, c )
      k = k + 1
#
#  Tabulate the exact and computed solutions.
#
  print ( '' )
  print ( '  I     X(I)      U(X(I))     Uh(X(I))' )
  print ( '' )
  for k in range ( 0, m * nel ):
    exact = dg1d_poisson_test_exact ( xh[k] )
    print ( '%2d  %8f  %8f  %8f' % ( k, xh[k], exact, uh[k] ) )
#
#  Evaluate the true solution at lots of points.
#
  x = np.linspace ( 0.0, 1.0, 101 )
  u = dg1d_poisson_test_exact ( x )
#
#  Make a plot comparing the exact and computed solutions.
#
  plt.plot ( xh, uh, label = 'approximate' )
  plt.plot ( x, u, label = 'exact' )
  plt.legend ( loc = 0 )
  plt.grid ( True )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---U(X)--->' )
  plt.title ( 'Compare computed and exact solutions' )
#
#  Save the graphics in a file.
#
  filename = f'./pic/n{nel}ss{ss}p{penal}.png'  
  plt.savefig ( filename )
  print ( '' )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.show ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'DG1D_POISSON_TEST' )
  print ( '  Normal end of execution.' )

  return

def dg1d_poisson_test_exact ( x ):
  import numpy as np
  value = ( 1.0 - x ) * np.exp ( - x ** 2 )
  return value

def dg1d_poisson_test_source ( x ):
  value = - ( 2.0 * x - 2.0 * ( 1.0 - 2.0 * x ) \
    + 4.0 * x * ( x - x ** 2 ) ) * np.exp ( - x * x )
  return value

def timestamp ( ):
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

if __name__ == '__main__':
  timestamp ( )
  dg1d_poisson_test ( )
  timestamp ( )
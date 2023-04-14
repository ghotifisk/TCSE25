! Page 412 Example 17.1

program hello
print *, 'Hello World!' ! My first Fortran program
end program


! Page 414 Example 17.2

program chessboard
  integer :: fieldno
  integer(kind=16) :: fieldval, sm = 0
  do fieldno = 1, 64 
    if (fieldno == 1) then
      fieldval = 1
    else 
      fieldval = 2 * fieldval
    end if 
    sm = sm + fieldval  
  end do
  print *, sm
end program 


! Page 414 Example 17.3

program collatz
  integer :: n = 100, nsteps = 0
  do while (n > 1)
    if (mod(n, 2) == 0) then
      n = n/2
    else 
      n = 3*n + 1
    end if
    nsteps = nsteps + 1    
  end do
  print *, 'reached 1 in', nsteps, 'steps'   
end program


! Page 415 Example 17.4

integer function fac(n); integer :: n
  integer :: i 
  fac = 1
  do i = 1, n
    fac = i * fac
  end do 
end function 
program facTest
  integer :: fac
  print *, fac(5)  ! Out: 120 
end program


! Page 416 Example 17.5

recursive function fac(n) result(res)
  integer :: n, res  ! type declaration for argument and result
  res = 1
  if (n == 1) return
  res = n * fac(n - 1)
end function
program facTest
  integer :: fac
  print *, fac(5)  ! Out: 120 
end program


! Page 416 Example 17.6

subroutine swap(x,y)
  integer :: x, y, temp
  temp = x; x = y; y = temp
end subroutine 
program testSwap
  integer :: a = 1, b = 2
  call swap(a,b)
  print *, a, b  ! Out: 2 1   
end program


! Page 417 Example 17.7

integer function square(n)
  integer :: n  ! if declared with attribute intent(in) ...
  n = n**2      ! ... then cought by compiler 
  square = n
end function
program testSquare
  integer square
  integer :: m = 2
  print *, square(m), m  ! Out:  4 4
end program


! Page 418 Example 17.8

real function integral(fun)
  interface
    real function fun(x); real :: x; end function     
  end interface
  integer, parameter :: n = 1000  ! constant
  real, parameter :: h = 1.0/n    ! constant
  integer :: i;  
  integral = 0.0
  do i = 0, n-1
    integral = integral + fun(h * (i + 0.5)) 
  end do
  integral = h * integral
end function
program integralTest
  real :: integral, res 
  res = integral(test_fun)  ! test_fun defined below in line 21
  print *, res              ! Out: 3.14159322
  contains
  real function test_fun(x)
    real :: x
    test_fun = 4.0/(1 + x*x)
  end function 
end program


! Page 420 Example 17.9

program eratos
  integer, parameter :: n = 10                          
  logical :: primes(2 : n) = .true.  ! array index starts at 2
  integer i, j  
  do i = 2, n
    do j = 2, n/i                    ! integer division n/i
      primes(i * j) = .false.        ! product cannot be prime
    end do
  end do  
  do i = 2, n 
    if (primes(i)) print *, i
  end do 
end program 


! Page 422 Example 17.10

module quicksort
  contains
  recursive function qsort(a) result(b)
    integer, dimension(:) :: a
    integer :: b(size(a))
    if (size(a) == 0) return	
    b = [ qsort( pack(a(2:), a(2:) < a(1)) ) ,   &   
          a(1),                                  &  ! pivot
          qsort( pack(a(2:), a(2:) >= a(1)) ) ]
  end function
end module
program quicksortTest
  use quicksort
  integer :: a(6) = [4,5,7,3,8,3]
  print '(*(i3))', qsort(a)  ! Out: 3  3  4  5  7  8
end program 


! Page 423 Example 17.11

program allocExample
  integer :: a(10) = [0,2,0,0,8,10,0,14,16,18]
  integer, dimension(:), allocatable :: arr
  integer :: m
  m = size(pack(a, a /= 0))
  allocate(arr(m))
  arr = pack(a, a /= 0)
  print '(*(i4))', arr  ! Out:  2   8  10  14  16  18
end program


! Page 425 Example 17.12

subroutine cgr(n, A, b, x)
  integer :: n; real :: A(n,n), b(n), x(n)
  real :: r(n), p(n), Ap(n), rs_old, alpha, rs_new; integer :: i 
  x = 0.0; r = b; p = b            ! start values
  rs_old = dot_product(r,r)        !  - dito -
  do i = 1, n
    Ap = matmul(A,p)
    alpha = rs_old / (dot_product(p, Ap))
    x = x + alpha * p
    r = r - alpha * Ap
    rs_new = dot_product(r,r) 
    if (sqrt(rs_new) < 1e-6) exit  ! sqrt built-in function
    p = r + (rs_new / rs_old) * p 
    rs_old = rs_new  
  end do 
end subroutine
program testGradient
  real :: A(4,4), b(4), x(4)
  A(1, :) = [ 9,   3,  -6,  12]
  A(2, :) = [ 3,  26,  -7, -11]
  A(3, :) = [-6,  -7,   9,   7]
  A(4, :) = [12, -11,   7,  65]
  b = [18, 11, 3, 73]
  call  cgr(4, A, b, x)  ! the result is stored in x
  print '(*(f7.3))', x   ! Out:  1.000  1.000  1.000  1.000
end program


! Page 426 Example 17.13

module laplace_det
  contains
  recursive function det(A) result(sum)
    real :: A(:,:), sum
    real :: A_sub( size(A,1)-1, size(A,1)-1 )
    integer :: col( size(A,1) ), n, j  
    n = size(A,1)
    if (n == 1) then 
      sum = A(1,1); return
    end if
    sum = 0.0
    col = [(j, j = 1, n)]
    do j = 1, n 
      A_sub = A(2:n, pack(col, col /= j))
      sum = sum + (-1)**(j+1) * A(1,j) * det(A_sub)
    end do   
  end function 
end module
program detTest
  use laplace_det
  real :: A(3,3) = reshape([1,2,3,1,1,1,3,3,1], [3,3], order=[2,1])
  print *, det(A)  ! Out: 2.00000000 
end program 


! Page 427 Example 17.14

program luTest
  real :: A(3,3) = reshape([1,2,3,1,1,1,3,3,1], [3,3], order=[2,1])
  integer, parameter :: n = size(A,1)
  real :: P(n,n) = 0, L(n,n) = 0, U(n,n) = 0, row(n)
  integer :: ipiv(n), info
  call sgetrf(n, n, A, n, ipiv, info)
  do i = 1, n
    L(i,  :i-1) = A(i,  :i-1);  L(i, i) = 1
    U(i, i:n)   = A(i, i:n ) 
    P(i,i) = 1
  end do
  do i = 1, n
    if (ipiv(i) /= i) then
      row = P(i,:); P(i,:) = P(ipiv(i),:); P(ipiv(i),:) = row 
    end if
  end do
  P = transpose(P)
  A = matmul(P, matmul(L,U))
  do i = 1, n
    print "(4(3f5.1'   '))", P(i,:), L(i,:), U(i,:), A(i,:)
  end do
end program 


! Page 430 Example 17.15

module fraction
  type fract
    integer :: num, den
  end type
  interface operator (+)
    procedure add_fract  
  end interface
  interface operator (==)
    procedure eq_fract
  end interface 
  contains
  function add_fract(a, b) result(c)
    type(fract), intent(in) :: a, b; type(fract) :: c  
    c % num = (a % num * b % den) + (b % num * a % den)
    c % den = a % den * b % den
  end function
  function eq_fract(a, b) result(eq)
    type(fract), intent(in) :: a, b; logical :: eq
    eq = (a % num * b % den == b % num * a % den)
  end function
end module
program fracTest
  use fraction
  type (fract) :: a = fract(3,4), b = fract(1,2), c, d
  logical :: eq
  c = a + b; d = b + a 
  eq = (c == d) 
  print '(2i3, l3)', c, eq  ! Out: 10 8 T
end program


! Page 431 Example 17.16

program sparseVectors
  integer, parameter :: n = 10; integer :: i, j, m
  integer :: a(n) = [0,2,0,0,8,10,0,14,16,18] 
  type node 
    integer :: idx, val 
  end type 
  type (node) :: v(n)
  type (node), allocatable :: w(:)
  v = [(node(i, a(i)), i = 1, n)]  ! implied do loop constructor
  m = size(pack(a, a /= 0))
  allocate(w(m))
  w = pack(v, v % val /= 0)
  do j=1, m
    print '(2i4)', w(j)
  end do
end program


! Page 432 Example 17.17

program pointerExample
  integer, pointer :: p, q
  integer, target :: a
  ! print *, p 
  p => null(); print *, p, associated(p)  ! Out: 0 F
  allocate(p); print *, p, associated(p)  ! Out: 0 T
  p = 1;       print *, p                 ! Out: 1
  q => p;      print *, q, associated(q)  ! Out: 1 T
  a = 2
  q => a;      print *, q                 ! Out: 2
  print *, p + a, p + q                   ! Out: 3 3
end program 


! Page 433 Example 17.18

program linkedList
  integer, parameter :: n = 10; integer :: i 
  integer :: a(n) = [0,2,0,0,8,10,0,14,16,18]
  type node
    integer :: index, value
    type (node), pointer :: next => null()  ! zero initialized 
  end type
  type (node), pointer :: root, current
  allocate(root)
  current => root
  do i = 1, n
    if (a(i) == 0) cycle  ! skip to next index
    current % index = i
    current % value = a(i)
    allocate(current % next)
    current => current % next 
  end do
  current => root   
  do while (associated(current % next))
    print *, current % index, current % value
    current => current % next
  end do     
end program


! Page 434 Example 17.19

program hello_mpi
  use mpi_f08
  integer :: rank, size, ierror
  call mpi_init(ierror) 
  call mpi_comm_size(mpi_comm_world, size, ierror)
  call mpi_comm_rank(mpi_comm_world, rank, ierror)
  print *, 'Hello World from process', rank, 'of',  size
  call mpi_finalize(ierror)
end program


! Page 435 Example 17.20

program hello_omp
  use omp_lib  ! Fortran module for parallel OpenMP execution
  call omp_set_num_threads(8)      
  !$omp parallel   
    print *, 'Hello from thread', omp_get_thread_num(), &
             'of', omp_get_num_threads()   
  !$omp end parallel
end program 


! Page 435 Example 17.21

real function f(x); real :: x 
  f = 4.0/(1.0 + x*x)
end function
function partial_pi(n, start, step) result(sum)
  integer :: n, start, step, i
  real :: h, sum, f 
  h = 1.0/n 
  !$omp parallel do reduction(+:sum)  
    do i = start, n-1, step  ! note step argument in loop
      sum = sum + h*f(h*(i+0.5))      
    end do
  !$omp end parallel do               
end function
program integralTest
  use mpi_f08
  real :: partial_pi, pi, pi_loc
  integer  ::  rank, size, ierror, tag, n
  call mpi_init(ierror)
  call mpi_comm_size(mpi_comm_world, size, ierror)
  call mpi_comm_rank(mpi_comm_world, rank, ierror)
  if (rank == 0)  n = 1000  
  call mpi_bcast(n, 1, mpi_int, 0, mpi_comm_world, ierror)
  pi_loc = partial_pi(n,rank,size)  ! OMP used here
  call mpi_reduce(pi_loc, pi, 1, mpi_float, mpi_sum, 0,  &
            mpi_comm_world, ierror)
  if (rank == 0) print *, pi  
  call mpi_finalize(ierror)
end program 


! Page 437 Example 17.23

program coArrayExample
  integer, codimension[*] :: hits = 0
  integer, parameter :: n = 10**6
  do i = 1, n/num_images()  ! assume n divisible by number of images
    call random_number(x); call random_number(y)
    if (x*x + y*y  <= 1.0) hits = hits + 1  ! local to each image
  end do
  sync all 
  if (this_image() == 1) then  ! executed only in image 1
    do i=2, num_images()       ! start at 2, collect from helpers
      hits = hits + hits[i]    ! collected
    end do
    print *, 4 * hits/real(n)  ! convert to real number
  end if 
end program


! Page 438 Nested Communication

module stencilComp
  contains
  subroutine stencil(U); real :: U(:,:)
    real, allocatable :: Tmp(:,:)
    integer :: n, m, i, j
    Tmp = U  ! automatic allocation
    n = size(U,1); m = size(U,2)
    do i = 2, n-1
      do j = 2, m-1
        Tmp(i,j) = 0.25 * (U(i-1,j) + U(i,j-1) + U(i+1,j) + U(i,j+1)) 
      end do 
    end do
    U = Tmp
  end subroutine
end module
program laplace
  use stencilComp 
  real :: U(10,10)  ! matrix to hold solution 
  real, codimension[*] :: coU(10,6)
  real :: x(10)
  integer :: i, j 
  if (this_image() == 1) then
    U = 0.0
    x = [(i * 1./9, i = 0, 9)]  ! implied do loop constructor
    U(1,:) = x * (1-x)          ! elementwise multiplication
    coU(:,:)    = U(:, :6)      ! local instance
    coU(:,:)[2] = U(:, 5:)      ! sent to helper image 2
  end if
  do i = 1, 40
    call stencil(coU)
    sync all
    if (this_image() == 1) coU(:, 1)[2] = coU(:, 5) 
    if (this_image() == 2) coU(:, 6)[1] = coU(:, 1)   
  end do
  if (this_image() == 1) then 
    U(:,  :5) = coU(:,  :5)
    U(:, 6: ) = coU(:, 2:)[2] 
  end if
  if (this_image() == 1) then 
    U = transpose(U)
    open(1, file="data.plt")
    do i = 1, 10
      do j = 1, 10
        write(1,*) i, j, U(i,j) 
      end do
      write(1,*) 
    end do
    close(1)
  end if
end program 












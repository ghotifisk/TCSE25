// Page 445 Example 18.2 Collatz Problem

var n = 100
while n > 1 {
  if n % 2 == 0 { n = n/2 }
  else { n = 3*n + 1 }
  // n = (n % 2 == 0) ? n/2 : 3*n + 1
  print(n) }
  
  
// Page 446 Example 18.3 Rice and Chessboard Problem
 
var grains: UInt = 0
var fieldval: UInt = 0
for fieldno in 1 ... 64 {
  fieldval = (fieldno == 1) ? 1 : 2*fieldval
  grains += fieldval }
print(grains)  // 18446744073709551615


// Page 446 Ranges

let range = 1 ..<  5
for i in range.reversed() { print(i, terminator:" ") }


// Page 446 Example 18.4 Monte Carlo Method

let samples = 100_000
var hits = 0
for _ in 1...samples {
  let x = Double.random(in: 0..<1)
  let y = Double.random(in: 0..<1)
  let d = x*x + y*y
  if d <= 1 { hits += 1 } }
print(4.0 * Double(hits) / Double(samples))  // 3.14844


// Page 447 Example 18.5 Factorial

func fac(n: Int) -> Int {
  var res = 1
  for i in 1...n { res *= i }
  return res }
print(fac(n: 5))

func fac(_ n: Int) -> Int
fac(5)


// Page 448 Example 18.6 Recursive Factorial

func fac(_ n: Int) -> Int { return (n == 1 ? 1 : n*fac(n-1)) }
print(fac(5))


// Page 448 Example 18.7

func swap(_ x: Int, _ y: Int) -> (Int, Int) { return (y, x) }
let (a, b) = swap(1, 2)
print(a, b)  // 2 1


// Page 448 Example 18.8

func swap_mod(_ a: inout Int, _ b: inout Int) {
  let tmp = a; a = b; b = tmp }
var a = 1, b = 2
swap_mod(&a, &b)
print(a, b)


// Page 449 Example 18.9

func ddx(_ f: (Double) -> Double, _ x: Double) -> Double {
  let h = 1e-14
  return (f(x+h) - f(x)) / h }
func testFun(_ x: Double) -> Double { return x*x }
print(ddx(testFun, 0.5))  // 0.9992007221626409


// Page 449 Example 18.10

typealias DtoD = (Double) -> Double
func ddx(_ f: @escaping DtoD) -> DtoD {
  func f_prime(_ x: Double) -> Double { return ddx(f, x) }
  return f_prime }
let testFun_prime = ddx(testFun)
print(testFun_prime(0.5))  // 0.9992007221626409


// Page 449 Operators

infix operator **
func **(lhs: Double, rhs: Int) -> Double {
  var res = 1.0
  for _ in 1...rhs { res *= lhs }
  return res }
print(4.2 ** 3)  // 74.08800000000001


// Page 449 Example 18.12

let pow = { (a: Int, b: Int) -> Int in
// let pow = { (a, b) in
  var res = 1
  for _ in 1...b { res *= a }
  return res }
print(pow(2, 4))  // 16

let pow = {var res = 1; for _ in 1...$1 {res *= $0}; return res}


// Page 449

print({$0 + $1}(1, 2))    // 3
print({$0 + $1}(1, 2.0))  // 3.0


// Page 452 Example 18.13 Sieve of Eratosthenes

let n = 100
var L = Array(2...n)
var P: [Int] = []
while L != [] {
  let p = L[0]
  P.append(p)
  for i in (0 ..< L.count).reversed() {
    if L[i] % p == 0 { L.remove(at: i) } } }
print(P)


// Page 454 Example 18.19 Horner’s Rule

let a = [1, 2, 3]; let x = 4
// var res = 0
// for c in a.reversed() { res = x*res + c }
let res_red = a.reversed().reduce(0) { x*$0 + $1 }


// Page 455 Example 18.20 Quicksort)

func qsort(_ lst: [Int]) -> [Int] {
  if lst == [] { return [] }
  let p = lst[0]
  let sml = lst[1...].filter { $0 < p }
  let grt = lst[1...].filter { $0 >= p }
  return qsort(sml) + [p] + qsort(grt) }
let lst = [4, 5, 7, 3, 8, 3]
print(qsort(lst))  // [3, 3, 4, 5, 7, 8]


// Page 455 Example 18.22 Remove Function

func remove(_ a: [Int], _ x: Int) -> [Int] {
  var b = a
  let i = b.firstIndex(of: x)
  if i != nil { b.remove(at: i!) }
  return b }
var a = [3, 1, 4, 1]
print(remove(a, 1))  // [3, 4, 1]
print(remove(a, 2))  // [3, 1, 4, 1]

// for x in L { if x % p == 0 { L = remove(L, x) } }
// if let i = i { b.remove(at: i) }
// if let i { b.remove(at: i) }


// Page 457 Example 18.23

struct Frac { var num, den: Int }
extension Frac {
  init(_ num: Int, _ den: Int){ self.num = num; self.den = den }
  func show() { print("\(self.num)/\(self.den)") }
  static func mul(_ a: Frac, _ b: Frac) -> Frac {
  // static func *(_ a: Frac, _ b: Frac) -> Frac {
    return Frac(a.num * b.num, a.den * b.den) } }
let a = Frac(3, 4)
let b = Frac.mul(a, a)
b.show()  // 9/16


// Page 458 Example 18.24

class Frac {
  var num, den: Int
  init(_ num: Int, _ den: Int){ self.num = num; self.den = den }
  func show() { print("\(self.num)/\(self.den)") }
  static func *(_ a: Frac, _ b: Frac) -> Frac {
           return Frac(a.num * b.num, a.den *  b.den) } }
let a = Frac(3, 4)
let b = a * a
b.show()  // 9/16
           
           
// Page 458 Example 18.25
           
class Pol {
  let coeff: [Double]
  init(_ coeff: [Double]) { self.coeff = coeff }
  func eval(_ x: Double) -> Double {
    return coeff.reversed().reduce(0) {x * $0 + $1} } }
extension Pol {
  static func +(_ ls: Pol, _ rs: Pol) -> Pol {
    let (hdg, ldg) =
      (ls.coeff.count < rs.coeff.count) ? (rs, ls) : (ls, rs)
    var sumCoeff = hdg.coeff
    for i in 0 ..< ldg.coeff.count { sumCoeff[i] += ldg.coeff[i] }
    return Pol(sumCoeff) } }
    
let p = Pol([1, 2, 3])
print(p.eval(4))  // 57.0
let q = Pol([4, 5])
let r = p + q     // p defined as above
print(r.coeff)    // [5.0, 7.0, 3.0]
print(r.eval(4))  // 81.0


// Page 458 Example 18.26 Subclassing

class Par: Pol {
  override init(_ coeff: [Double]) {
    if coeff.count != 3 { fatalError("No parabola") }
    super.init(coeff) }
  override func eval(_ x: Double) -> Double {
    return coeff[0] + coeff[1]*x + coeff[2]*x*x }
  static func +(ls: Par, rs: Par) -> Par {
    var sumCoeff: [Double] = [0, 0, 0]
    for i in 0...2 { sumCoeff[i] += ls.coeff[i] + rs.coeff[i] }
    return Par(sumCoeff) } }  // end subclass Par
let p = Par([1, 2, 3]); let q = Par([3, 2, 1])
print((p + q).eval(4))  // 84.0


// Page 460 Example 18.27 Linked List

class Node {
  var val: Int
  var next: Node?  // nil-initiallzed
  init(_ val: Int) { self.val = val } }
class LinkedList {
  var head, tail: Node?
  func addNode(_ val: Int) {
    let newNode = Node(val)
    if tail == nil { head = newNode }
    else { tail!.next = newNode }
    tail = newNode }
  func show() {
    var node = head
    while node != nil {
      print(node!.val, terminator: " ")
      node = node!.next } }
} // end linked list def

var list = LinkedList()
for i in 1...10 { list.addNode(i) }
list.show()  // 1 2 3 4 5 6 7 8 9 10


// Page 461 Linear Algebra

import Accelerate
typealias Vec = [Double]


// Page 462 Scalar Multiplication

func *(_ a: Double, _ x: Vec) -> Vec {
  let n = x.count
  var res = x
  cblas_dscal(n, a, &res, 1)
  return res }


// Page 462 Vector Addition

func +(_ x: Vec, _ y: Vec) -> Vec {
  let n = x.count
  var res = y
  cblas_daxpy(n, 1, x, 1, &res, 1)
  return res }


// Page 463 Vector Subtraction
 
func -(_ x: Vec, _ y: Vec) -> Vec {
  let n = x.count
  var res = x
  cblas_daxpy(n, -1, y, 1, &res, 1)
  return res }


// Page 463 Vector Dot Product

func *(_ x: Vec, _ y: Vec) -> Double {
  let n = x.count
  return cblas_ddot(n, x, 1, y, 1) }

let x: Vec = [1, 2, 3]; let y: Vec = [3, 2, 1]
print(2 * x)  // [2.0, 4.0, 6.0]
print(x + y)  // [4.0, 4.0, 4.0]
print(x - y)  // [-2.0, 0.0, 2.0]
print(x * y)  // 10.0


// Page 463 Matrices

typealias Mat = [[Double]]  // equivalent = [Vec]

extension Mat {
  func show(_ n: Int, _ m: Int) {
    for i in 0 ..< self.count {
      for j in 0 ..< self[0].count {
        print(String(format: "%\(n+m+3).\(m)f",
                              self[i][j]), terminator:" ") }
      print() } } }


// Page 463 Flatten and Reshape

func flatten(_ A: Mat) -> (Vec, Int, Int) {
  let rows = A.count; let cols = A[0].count
  var data: Vec = []
  for j in 0 ..< cols { for i in 0 ..< rows {
    data.append(A[i][j]) } }
  return (data, rows, cols) }

func reshape(_ data: Vec, _ rows: Int, _ cols: Int) -> Mat  {
  var A: Mat = []
  for i in 0 ..< rows {
    var w: Vec = []
    for j in 0 ..< cols { w.append(data[i + rows * j]) }
    A.append(w) }
  return A }

let A: Mat = [[1, 2, 3], [4, 5, 6]]
A.show(1, 2)
let (data, rows, cols) = flatten(A)  // A as above
print(data, rows, cols)  // [1.0, 4.0, 2.0, 5.0, 3.0, 6.0] 2 3
let B = reshape(data, rows, cols)  // values from example above
print(B)  // [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


// Page 464 Matrix-Vector Multiplication

func *(_ A: Mat, x: Vec) -> Vec {
  let (data, rows, cols) = flatten(A)
  var res = Vec(repeatElement(0, count: rows))
  cblas_dgemv( CblasColMajor, CblasNoTrans,
    rows, cols, 1.0, data, rows, x, 1, 0.0, &res, 1 )
  return res }


// Page 465 Matrix-Matrix Multiplication

func *(A: Mat, B: Mat) -> Mat {
  let (dataA, rowsA, _) = flatten(A)
  let (dataB, rowsB, colsB) = flatten(B)
  var dataC = Vec(repeating: 0.0, count: rowsA * colsB)
  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
    rowsA, colsB, rowsB,
    1.0, dataA, rowsA, dataB, rowsB, 1.0, &dataC, rowsA )
  return reshape(dataC, rowsA, colsB) }

let A: Mat = [[1, 2, 3], [4, 5, 6]]
let x: Vec = [3, 2, 1]
print(A * x)  // [10.0, 28.0]
let B: Mat = [[1,4], [2,5], [3,6]]
print(A * B)  // [[14.0, 32.0], [32.0, 77.0]]


// Page 465 Example 18.28 Matrix Inversion

func inv(_ A: Mat) -> Mat {
  var (data, rows, cols) = flatten(A)
  var info = 0; var ipiv = [Int](repeatElement(0, count: rows))
  dgetrf_(&rows, &cols, &data, &rows, &ipiv, &info)
  var lwork = rows; var work = Vec(repeatElement(0, count: rows))
  dgetri_(&rows, &data, &rows, &ipiv, &work, &lwork, &info)
  return reshape(data, rows, cols) }

let A: Mat = [[1, 2, 3], [1, 1, 1], [3, 3, 1]]
inv(A).show(1, 2)


// Page 467 Gauss-Seidel Method

let A: Mat = [[4, 3, 0], [3, 4, -1], [0, -1, 4]]
var L = A; var U = A
for i in 0..<3 { for j in 0..<3 {
  if  i < j { L[i][j] = 0 }
  else { U[i][j] = 0 } } }

let Linv = inv(L)
Linv.show(1, 3)

var x: Vec = [3, 3, 3]
var b: Vec = [24, 30, -24]

for _ in 1...10 { x = Linv * (b - (U * x)) }
print(x)      // [2.97, 4.02, -4.99]    (rounded)
print(A * x)  // [23.96, 30.00, -24.0]  (rounded)

var (data, rows, _) = flatten(L)
var info = 0
dtrtri_("l", "n", &rows, &data, &rows, &info)
let LtriInv = reshape(data, 3, 3)
LtriInv.show(1, 3)


// Page 468 Linear Equations: Direct Solution

func solve(_ A: Mat, _ b: Vec) -> Vec {
  var (data, rows, _) = flatten(A)
  var nrhs = 1; var ipiv = [Int](repeatElement(0, count: rows))
  var bx = b; var info = 0
  dgesv_(&rows, &nrhs, &data, &rows, &ipiv, &bx, &rows, &info)
  return bx }

let A: Mat = [[4, 3, 0], [3, 4, -1], [0, -1, 4]]
let b: Vec = [24, 30, -24]
print(solve(A, b))


 



 



include("../src/mpgmres.jl");

# TEST_MPGMRES Test example for MPGMRES.
# 
# This file generates Example 5.1 from 
# <a href="http://www.cs.ubc.ca/~tyronere/TR-2011-12.pdf">Multi-preconditioned GMRES</a>.
#
# This takes a normally distributed random matrix A with two
# preconditioners which are also random matrices.  The right hand side is
# chosen such that it's an eigenvector of A^{-1}P_2A^{-1}P_1, and so the
# exact solution lies in the multi-Krylov space in which full-MPGMRES looks
# for its iterates after two iterations.
#
# If we choose truncated MPGMRES with alternating preconditioners, then the
# exact solution is also in the approximation space after two iterations
# here.  This case is included as a second example, which illustrates the
# use of function handles.
#
# See also:
# MPGMRES

n = 100;
println("Generating random $(n)x$(n) matrices and right hand side vector...")
A = randn(n,n);
P1 = randn(size(A));
P2 = randn(size(A));
PM = (A/P2)*(A/P1);
L,B = eigen(PM);
L = diagm(L);

b = B[:,end]; # note: b could be complex, even though A is real
# Setting up for MPGMRES, used mostly with defaults:
P = [P1,P2]; # preconditioners must be passed to mpgmres as a cell array 
println("Solving with full MPGMRES...")
x,relres,iter,resvec = mpgmres(A,b,P,type=TypeMP(type="full"));
if iter < 3
    println("full MPGMRES converged sucessfully in $(iter) iterations to a relative residual of norm $(relres)!")
else
    println("full MPGMRES converged sucessfully in $(iter) iterations to a relative residual of norm $(relres)!")
    println("This is less than the expected number of iterations for the FULL method. Run again to check if this is a problem with the random matrices used")
end

# Do a second run of truncated MPGMRES:
# set up anonymous functions
A_an(u) = A*u;
P1_an(u) = P1\u;
P2_an(u) = P2\u;
P_an = [P1_an,P2_an];
# set 'type' structure
type = TypeMP(type="trunc", col=true, method="alternate")
println("Solving with truncated MPGMRES...")
x,relres,iter,resvec = mpgmres(A_an,b,P_an,type=type);
if iter < 3
    println("full MPGMRES converged sucessfully in $(iter) iterations to a relative residual of norm $(relres)!")
else
    println("full MPGMRES converged sucessfully in $(iter) iterations to a relative residual of norm $(relres)!")
    println("This is less than the expected number of iterations for the FULL method. Run again to check if this is a problem with the random matrices used")
end
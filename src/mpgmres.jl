using LinearAlgebra

include("utils.jl")

function fullmultiprecondition(pre,Q)
    lpre = length(pre);
    [lA,lQ] = size(Q);
    z = zeros(Float64, (lA,lQ*lpre));
    ind = 1;
    ivec = 1:lQ*lpre;
    for i=1:lpre
        if isa(pre[1],Function)
            for j = 1:lQ
                z[:,ivec(ind)]=pre[i](Q[:,j]);
                ind = ind+1;
            end
        else
            z[:,ivec[ind:ind+lQ-1]]=pre[i]\Q;
            ind = ind + lQ;
        end
    end
    return z
end

function mpgmres(A,b,P;type=TypeMP(),tol=[1e-6,0],maxit=length(b),x0=zeros(Float64,size(b)),store_Z=false,test_orth=false,save_mats=false)
    # make sure b is a vector
    if isa(b,Vector) == false
        error("Right hand side b must be a vector")
    end
    lb = length(b);

    # get type of A
    if isa(A,Array)
        lA,wA = size(A);
        if lA ~= wA
            error("The input matrix A must be square")
        end    
        if lb != lA
            error("The right hand side vector must be of the same length as A")
        end
        Atype = "matrix";
    elseif isa(A,Function)
        Atype = "func";
    else
        error("Unsupported type of A supplied")
    end

    # validate tol is a vector with 2 entries
    if isa(tol, Float64)
        tol = [tol 0]
    elseif isa(tol, Vector)
        ltol = length(tol);
        if ltol == 0
            tol = [1e-6, 0]
        elseif ltol == 1
            tol = [tol[1], 0]
        elseif ltol > 2
            error("The tolerance must be a vector with at most two entries")
        end
    else
        error("The tolerance must be a numeric value or vector with at most two entries")
    end

    if minimim(tol) < 0
        error("Tolerance values should be non-negative")
    elseif maximum(tol) <= 0
        error("At least one tolerance value should be strictly positive")
    end

    if maxit >= 10000
        @warn "maxits was unspecified and will be set to the length of b. For large problem sizes specify maxit to avoid allocation of a large amount of memory"
    end

    # validate x0
    if isa(x0,Vector)
        lx0 = length(x0)
        if lx0 != lb
            error("The starting vector must be of the same length as b")
        end
    else
        error("Initial guess x0 must be a vector")
    end

    small = 10*eps();

    if isa(P, Vector)
        k = length(P)
    else
        k = 1;
        P = [P]
    end


    if type.type == "trunc"
        nmaxits = maxits*k;
    else # "full"
        if k == 1
            nmaxits = maxits;
        else
            nmaxits = (maxits^(k+1) - maxits)/(maxits - 1);
        end
    end

    indvar = min(lb,nmaxits);
    if store_Z
        Z = zeros(Float64, (lb,indvar));
    else
        #######################################################################
        #!! We don't need to store Z, as Z_k = [P_1^{-1} V_k ... P_t^{-1} V_k]#
        #!! and hence (Z^)y_k = P_1^{-1}(V^(:,Vindex_1))y_k(yindex_1) + ...   #
        #!!                     P_t^{-1}(V^(:,Vindex_t))y_k(yindex_t)         #
        #!! the values for the indices are stored in                          #
        #!! V_index and yk_index                                              #
        #######################################################################

        end_V = Array{Any}(undef, (k,1)); # similar to Matlab's cell array
        end_V[:] .= 0;
        Zinfo = {"yk_index": zeros(nmaxits,1), "V_index": zeros(nmaxits,1), "end_V": end_V}
        lastV = 0
    end

    V = zeros(Float64, (lb,indvar+1));
    H = zeros(Float64, (indvar+1,indvar));
    resvec = zeros(Float64, indvar);
    mvecs = zeros(Float64, indvar);
    c = zeros(Float64, indvar);
    s = zeros(Float64, indvar);
    rhs = zeros(Float64, indvar);

    lindep_flag = 0;    # set the linear dependence flag - set to 1 if H_(p+1,p) = 0

    r = Atype=="matrix" ? b-A*x0 : b-A(x0);           # initial residual
    

    mvecs[1] = mvecs[2] = 1.0;
    nr = norm(r);
    if nr <= max(tol[2],0) # check for convergence in absolute residual norm
        println("MPGMRES converged immediately due to absolute tolerance!")
        x = x0;
        relres = 1;
        resvec[1] = 1.0;
        iter = 0;
        return x,relres,iter,resvec,mvecs
    end

    rhs[1] = nr;
    resvec[1] = 1;
    V[:,1] .= r/nr;
    Z_temp = fullmultiprecondition(P,V[:,1]);       # send to multiprecondition to get Z_temp
    z_it = [1,0];                                   # [(outer) iteration, col of Z]
    Zk = k;                                         # a parameter which stores the size of the current Z_i

    if !store_Z
        VforZ = 1;
        ind_V = ones(Float64, (k,1));
    end

    pp = 0;     # initialize a concurrent index
    nVk = 0;    # initialize the current size of Vk

    for p=1:nmaxits         # loop over the columns of Zk
        pp = pp+1;
        w = Atype == "matrix" ? A*Z_temp[:,z_it[2]+1] : A(Z_temp[:,z_it[2]+1]); # initial residual
        mvecs[z_it[1]+1] = mvecs[z_it[1]+1] + 1;
        for i = 1:pp                  # loop over columns of V
            H[i,pp] = V[:,i]'*w;
            w = w - H[i,pp]*V[:,i];   # remove component of V_i from w
        end
        nw = norm(w);
        H[pp+1,pp] = nw;          # calculate norm of vector w

        # TEST if w is the zero vector, hence Z_k is linearly dependent
        if H[pp+1,pp] < SMALL
             lindep_flag = 1
        end
        for l=1:pp-1  # update previous rots
            h1 = H[l,pp];
            h2 = H[l+1,pp];
            H[l,pp] = conj(c[l])*h1 + conj(s[l])*h2;
            H[l+1,pp] = -s[l]*h1 + c[l]*h2;
        end
        h1 = H[pp,pp];
        h2 = H[pp+1,pp];
        gam = sqrt(abs(h1)^2 + h2^2);
        c[pp] = h1/gam;
        s[pp] = h2/gam;
        H[pp,pp] = conj(c[pp])*h1+conj(s[pp])*h2;
        H[pp+1,pp] = 0;

        # update rhs
        rhs_old = rhs;              # save old rhs
        rhs[pp+1] = -s[pp]*rhs[pp];
        rhs[pp] = conj(c[pp])*rhs[pp];

    end

end
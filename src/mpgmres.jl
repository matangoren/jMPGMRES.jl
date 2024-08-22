using LinearAlgebra
using Random
using MAT

include("utils.jl")

function fullmultiprecondition(pre,Q)
    lpre = length(pre);
    lQ = 1
    if length(size(Q)) == 1
        lA = size(Q,1)
    else
        lA,lQ = size(Q)
    end
    z = zeros(ComplexF64, (lA,lQ*lpre));
    ind = 1;
    ivec = 1:lQ*lpre;
    for i=1:lpre
        if isa(pre[1],Function)
            for j = 1:lQ
                z[:,ivec[ind]]=pre[i](Q[:,j]);
                ind = ind+1;
            end
        else
            z[:,ivec[ind:ind+lQ-1]]=pre[i]\Q;
            ind = ind + lQ;
        end
    end
    return z
end

function multipreconditionvec(pre,Q)
    n,m = size(Q);
    k = length(pre);
    z = zeros(ComplexF64, (n,k));
    for i=1:m
        if isa(pre[i],Function)
            z[:,i] = pre[i](Q[:,i]);
        else
            z[:,i] = pre[i]\Q[:,i];
        end
    end
    for i = m+1 : length(pre)
        if isa(pre{i},Function)
            z[:,i] = pre[i](Q[:,mod(i,m)+1]);
        else
            z[:,i] = pre[i]\Q[:,mod(i,m)+1];
        end
    end
    return z
end

function increase_ind_V(ind_V,k)
    sizeInd_V = length(ind_V);
    if sizeInd_V < k
        new_ind_V = zeros(k);
        new_ind_V[1:sizeInd_V] = ind_V;
        for jv = sizeInd_V + 1:k
            new_ind_V[jv] = ind_V[mod(jv,sizeInd_V)+1];
        end
    else
        new_ind_V = ind_V;
    end
    return new_ind_V
end

function mpgmres(A,b,P;type=TypeMP(),tol=[1e-6,0],maxits=length(b),x0=zeros(Float64,size(b)),store_Z=false,test_orth=false,save_mats=false)
    # make sure b is a vector
    if isa(b,Vector) == false
        error("Right hand side b must be a vector")
    end
    lb = length(b);

    # get type of A
    if isa(A,Array)
        lA,wA = size(A);
        if lA != wA
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

    if minimum(tol) < 0
        error("Tolerance values should be non-negative")
    elseif maximum(tol) <= 0
        error("At least one tolerance value should be strictly positive")
    end

    if maxits >= 10000
        @warn "maxits was unspecified and will be set to the length of b. For large problem sizes specify maxits to avoid allocation of a large amount of memory"
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

    SMALL = 10*eps();

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

    nmaxits = Int64(nmaxits);

    indvar = min(lb,nmaxits);
    if store_Z
        Z = zeros(ComplexF64, (lb,indvar));
    else
        #######################################################################
        #!! We don't need to store Z, as Z_k = [P_1^{-1} V_k ... P_t^{-1} V_k]#
        #!! and hence (Z^)y_k = P_1^{-1}(V^(:,Vindex_1))y_k(yindex_1) + ...   #
        #!!                     P_t^{-1}(V^(:,Vindex_t))y_k(yindex_t)         #
        #!! the values for the indices are stored in                          #
        #!! V_index and yk_index                                              #
        #######################################################################

        # end_V = Array{Any}(undef, (k,1)); # similar to Matlab's cell array
        # end_V[:] .= 0;
        # Zinfo = {"yk_index": zeros(nmaxits,1), "V_index": zeros(nmaxits,1), "end_V": end_V}
        Zinfo = [ZinfoInstance(nmaxits,0) for _=1:k]
        lastV = 0
    end

    V = zeros(ComplexF64, (lb,indvar+1));
    H = zeros(Float64, (indvar+1,indvar));
    resvec = zeros(Float64, indvar);
    mvecs = zeros(Float64, indvar);
    c = zeros(Float64, indvar);
    s = zeros(Float64, indvar);
    rhs = zeros(ComplexF64, indvar);

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
        ind_V = ones(Int64, k);
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

        # test_lindep_block;
        if (lindep_flag==1)&&((abs(rhs[pp+1]) >= max(tol[1]*nr,SMALL))||(isnan(abs(rhs[pp+1]))))&&(nVk>1) # the last column of Z is dependent on the others
            println("Column of Z linearly dependent...removing")
            pp = pp-1;          # reduce index by 1
            lindep_flag = 0;    # reset linear dependence flag
            rhs = rhs_old;      # replace rhs
        else
            nVk = nVk + 1;
            if store_Z
                Z[:,pp] = Z_temp[:,z_it[2]+1];
            else
                ik = Int64(ceil(nVk/VforZ)); # find which preconditioner
                Zinfo[ik].end_V += 1;
                Zinfo[ik].yk_index[Zinfo[ik].end_V] = pp;
                if isa(Zinfo[ik].V_index,Cell) # then type.col == 1, and cells are involved
                    Zinfo[ik].V_index[Zinfo[ik].end_V] = ind_V;
                    Zinfo[ik].wt[Zinfo[ik].end_V] = wt;
                    lastV = max(lastV,maximum(Zinfo[ik].V_index[Zinfo[ik].end_V]));
                else
                    Zinfo[ik].V_index[Zinfo[ik].end_V] = ind_V[nVk];
                    lastV = max(lastV,Zinfo[ik].V_index[Zinfo[ik].end_V]);
                end

            end
            if lindep_flag==1  # we have a lucky breakdown
                z_it[1] += 1;
                resvec[z_it[1]] = abs(rhs[pp+1])/nr;    # save residual to resvec
                println("MPGMRES converged -- lucky breakdown!")
                break
            end
            V[:,pp+1] = w/nw;       # set new basis vector V_{p+1}
            if test_orth            # test if V's are still orthogonal (see note above)
                S = triu(V'*V,1); 
                println("Measure of loss of orthog.")
                # ADD print of norm...?
                # norm((eye(size(S)) + S)\S)
            end
        end
        z_it[2] += 1;         # update the index column of Z we're working on
        
        if z_it[2] == Zk
            resvec[z_it[1]+1] = abs(rhs[pp+1])/nr;    # save residual to resvec
            
            # test convergence
            if resvec[z_it[1]+1] <= max(tol[1],tol[2]/nr)
                println("MPGMRES converged!")
                z_it[1] += 1;
                break
            end
            # multiprecondition
            if nVk == 0
                error("All current search directions are linearly dependent on the previous ones. Re-run with a different truncation rule or starting vector")
            end

            if type.type == "full"
                ind_V = repeat(pp+1-nVk+1:pp+1,k);
                Z_temp = fullmultiprecondition(P,V[:,pp+1-nVk+1:pp+1]);
                Zk = k*nVk;
                if ~store_Z
                    VforZ = nVk;
                end
                nVk = 0; # reset nV
            else # type.type == "trunc"
                if type.col
                    if type.method == "inorder" # P_i^{-1}Ve_i
                        ind_V = pp+1 .+ (1-nVk:0);
                        Z_temp = multipreconditionvec(P,V[:,ind_V]);
                        if ~store_Z
                            VforZ = 1;
                            ind_V = increase_ind_V(ind_V,k);
                        end
                    elseif type.method == "reverse" # % P_i^{-1}Ve_j, j = n-i+1
                        ind_V = pp+1 .+ (0:-1:1-nVk) # fliplr(pp+1 + (1-nVk:0));
                        Z_temp = multipreconditionvec(P,V[:,ind_V]);
                        if ~store_Z
                            VforZ = 1;
                            ind_V = increase_ind_V(ind_V,k);
                        end
                    elseif type.method == "alternate" # swaps between previous two cases
                        if rem(pp,2) == 0
                            ind_V = (pp+1):-1:(pp+1-nVk+1) # fliplr(pp+1-nVk+1:pp+1);
                            Z_temp = multipreconditionvec(P,V[:,ind_V]);
                            if ~store_Z
                                ind_V = increase_ind_V(ind_V,k);
                            end
                        else
                            ind_V = pp+1-nVk+1:pp+1;
                            Z_temp = multipreconditionvec(P,V[:,ind_V]);
                            if ~store_Z
                                VforZ = 1;
                                ind_V = increase_ind_V(ind_V,k);
                            end
                        end
                    elseif type.method == "random" # P_i^{-1}Ve_j, j random
                        order = randperm(nVk);
                        ind_V = zeros(nVk);
                        for iv = 1:nVk
                            ind_V[iv] = pp+1-nVk+order[iv];
                        end
                        Z_temp = multipreconditionvec(P,V[:,ind_V]);
                        if ~store_Z
                            VforZ = 1;
                            ind_V = increase_ind_V(ind_V,k);
                        end
                    else
                        error("Unsupported value of type.method")  
                    end
                    Zk = k; nVk = 0;      # update size of Z_k and current size of V_k
                else # type.col = False
                    if (z_it[1] == 1) && (~store_Z) # update V_index so it's a cell array, not a vector                    
                        for ii = 1:k
                            Zinfo[ii].V_index = Cell(nmaxits);
                            Zinfo[ii].V_index[1] = 1;
                            Zinfo[ii].wt = Cell(nmaxits);
                            Zinfo[ii].wt[1] = 1;
                        end
                    end
                    if type.method == "sum" # sum the columns
                        VforZ = 1;
                        ind_V = pp+1-nVk+1:pp+1;
                        wt = ones(nVk);
                        v_temp = V[:,ind_V]*wt;
                        Z_temp = fullmultiprecondition(P,v_temp);
                    elseif type.method == "random" # sum the columns with random weights
                        ind_V = pp+1-nVk+1:pp+1;
                        wt = rand(nVk);
                        v_temp = V[:,ind_V]*wt;
                        Z_temp = fullmultiprecondition(P,v_temp);
                    else
                        error("Unsupported value of type.method") 
                    end
                    Zk = k; nVk = 0;      # update size of Z_k and current size of V_k
                end
            end
            z_it[1] += 1;
            mvecs[z_it[1]+1]= mvecs[z_it[1]];
            z_it[2] = 0;    
        end

        if save_mats
            matwrite("mpgmres_const.mat",Dict("V" => V, "H" => H, "Z" => Z));
        end
    
    end

    if save_mats
        matwrite("mpgmres_b_const.mat",Dict("V" => V, "H" => H, "Z" => Z));
    end
    yk = H[1:pp,1:pp]\rhs[1:pp];
    if store_Z
        x = x0 + Z[:,1:pp]*yk;
    else
        x = x0;
        if isa(Zinfo[1].V_index,Cell) # then type.col = 0
            for ii = 1:k
                Vyk_i = zeros(lb);
                for iv = 1:Zinfo[ii].end_V # get the summed matrix
                    Vyk_i = Vyk_i + (V[:,Zinfo[ii].V_index[iv]]*Zinfo[ii].wt[iv])*yk[Zinfo[ii].yk_index[iv]];
                end
                if isa(P[ii],Function)
                    x += P[ii](Vyk_i);
                else
                    x += P[ii]\Vyk_i;
                end
                
            end
        else
            for ii = 1:k
                Vyk_i = V[:,Zinfo[ii].V_index[1:Zinfo[ii].end_V]]*yk[Zinfo[ii].yk_index[1:Zinfo[ii].end_V]];
                if isa(P[ii],Function)
                    x += P[ii](Vyk_i);
                else
                    x += P[ii]\Vyk_i;
                end
            end
        end
    end

    resvec = resvec[1:z_it[1]];
    mvecs = mvecs[1:z_it[1]];
    iter = length(resvec) - 1;
    relres = resvec[end];

    return x,relres,iter,resvec,mvecs
end
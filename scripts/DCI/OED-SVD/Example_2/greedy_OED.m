% Script to compute ESE and ESK and perform greedy OED
% Note that the datasets loaded here are slightly different from the ones
% used in the paper.  These used a coarser mesh.  This gives a slightly
% different sequence of OED than the finer mesh, but all of the results
% are within sampling errors.

% Use a forward finite difference approximation, so load the values and
% perturbations

load combined_dp1.dat
load combined_dp2.dat
load combined_dp3.dat
load combined_dp4.dat
load combined_dp5.dat
load combined_dp6.dat
load combined_dp7.dat
load combined_dp8.dat
load combined_dp9.dat

N = size(combined_dp1,1);
M = size(combined_dp1,2);

ESE = zeros(N,1);

% Compute first design using just ESE

for k=1:N
    currESE = 0.0;
    for m=1:M
        J = [combined_dp1(k,m) combined_dp2(k,m) combined_dp3(k,m) combined_dp4(k,m) combined_dp5(k,m) ...
            combined_dp6(k,m) combined_dp7(k,m) combined_dp8(k,m) combined_dp9(k,m)];
        [~,S,~] = svd(J);
        d = S(1);
        if size(S,1) == 1
            d = S(1);
        else
            d = prod(diag(S));
        end

        currESE = currESE + 1/M*d;

    end
    if (mod(k,100) == 0)
        k
    end
    ESE(k,1) = currESE;
end

[~,design] = max(ESE);

% Compute subsequent OED using ESK
ESK = zeros(N,1);

for iter = 1:2
    for k = 1:N

        currESK = 0.0;

        for m = 1:M
            J = [];
            for j=1:length(design)
                J = [J; combined_dp1(design(j),m) combined_dp2(design(j),m) combined_dp3(design(j),m) combined_dp4(design(j),m) combined_dp5(design(j),m) ...
                    combined_dp6(design(j),m) combined_dp7(design(j),m) combined_dp8(design(j),m) combined_dp9(design(j),m)];
            end
            J = [J; combined_dp1(k,m) combined_dp2(k,m) combined_dp3(k,m) combined_dp4(k,m) combined_dp5(k,m) ...
                combined_dp6(k,m) combined_dp7(k,m) combined_dp8(k,m) combined_dp9(k,m)];
            [~,S,~] = svd(J);
            if size(S,1) == 1
                d = S(1);
            else
                d = prod(diag(S));
            end


            mxskew = 0.0;
            for n=1:size(J,1)
                Jhat = J;
                Jhat(n,:) = [];
                jnorm = norm(J(n,:));
                [~,Shat,~] = svd(Jhat);
                if size(Shat,1) == 1
                    dhat = Shat(1);
                else
                    dhat = prod(diag(Shat));
                end

                mxskew = max(mxskew, jnorm*dhat/d);
            end
            currESK = currESK + 1/M*1/mxskew;
        end

        ESK(k,1) = currESK;


    end
    
    [~,newdes] = max(ESK);
    design = [design newdes];
    x = linspace(0,1,61);
    y = x;
    [X,Y] = meshgrid(x,y);

    % Unfortunately, the mesh generator in MrHyDE is not the same as
    % meshgrid - we just need to re-sort the first two rows
    data = reshape(ESK,61,61);
    for j=1:30
        data(2*j-1,1) = ESK(4*j-3,1);
        data(2*j,1) = ESK(4*j-2,1);
        data(2*j-1,2) = ESK(4*j,1);
        data(2*j,2) = ESK(4*j-1,1);
    end
    figure
    surf(X,Y,data,'linestyle','none','facecolor','interp','facelighting','phong'); colorbar; view(0,90)

end

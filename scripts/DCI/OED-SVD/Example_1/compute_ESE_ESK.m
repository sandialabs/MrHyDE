% Script to compute ESE and ESK

% Use a forward finite difference approximation, so load the values and
% perturbations

load combined_dp1.dat
load combined_dp2.dat

N = size(combined_dp1,1);
M = size(combined_dp1,2);

ESE = zeros(N,N);
ESK = zeros(N,N);

for k=1:N
    for j=1:N
        currESE = 0.0;
        currESK = 0.0;
        for m=1:M
            J = [combined_dp1(k,m) combined_dp2(k,m);
                combined_dp1(j,m) combined_dp2(j,m)];
            [~,S,~] = svd(J);
            d = prod(diag(S));
            currESE = currESE + 1/M*d;

            mxskew = 0.0;
            for n=1:size(J,1)
                Jhat = J;
                Jhat(n,:) = [];
                jnorm = norm(J(n,:));
                [~,Shat,~] = svd(Jhat);
                mxskew = max(mxskew, jnorm*Shat(1,1)/d);
            end
            currESK = currESK + 1/M*1/mxskew;
        end
        ESE(k,j) = currESE;
        ESK(k,j) = currESK;
        

    end
    k
end
x = linspace(0,1,41);
y = x;
[X,Y] = meshgrid(x,y);

figure
surf(X,Y,ESE,'linestyle','none','facecolor','interp','facelighting','phong'); colorbar; view(0,90)

figure
surf(X,Y,ESK,'linestyle','none','facecolor','interp','facelighting','phong'); colorbar; view(0,90)
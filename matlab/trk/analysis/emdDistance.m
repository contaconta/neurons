function d = emdDistance(hist1, hist2)


N= length(hist1);
THRESHOLD= 5;
extra_mass_penalty= -1;
flowType= 1;
D= ones(N,N).*THRESHOLD;
for i=1:N
    for j=max([1 i-THRESHOLD+1]):min([N i+THRESHOLD-1])
        D(i,j)= abs(i-j); 
    end
end

d = emd_hat_gd_metric_mex(hist1', hist2', D, extra_mass_penalty, flowType);

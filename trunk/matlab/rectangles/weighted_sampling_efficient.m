function samp = weighted_sampling_efficient(data, w, N)

data = data(:);

%inds = [];
samp = [];
n = 0;


while length(samp) < N
    i = randsample(length(data), N, true, w);
    
    i = unique(i);
    
    samp = [samp; data(i)]; %#ok<AGROW>
    
    w(i) = 0;
    n = n + 1;
end

disp(['   required ' num2str(n) ' iterations.']);
samp = samp(1:N);
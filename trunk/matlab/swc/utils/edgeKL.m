function KL = edgeKL(Iraw, pixelList, G0, T)

cls = class(Iraw);
BINS = 15;
Iraw = double(Iraw);
eta = .000000001;   % very small number to handle inf's and NaN's in KL div

% only operate on the upper triangle
G0 = triu(G0);

% find the locations of the edges in G0
[r,c] = find(G0);

bins = 0:(double(intmax(cls))/(BINS-1)):double(intmax(cls));

% initialize the KL matrix to all zeros
KL = sparse([],[],[], size(G0,1), size(G0,1),0); H = zeros(size(G0,1),BINS);
for i = 1:size(G0,1)
    patch = Iraw(pixelList{i});
    hraw = histc(patch,bins);
    hraw = hraw + eta;
    H(i,:) = hraw/sum(hraw);
end

KLIST = zeros([1 length(r)]);

% loop through each edge
for i = 1:length(r)
    
    H1 = H(r(i),:);
    H2 = H(c(i),:);
    
    KL(r(i),c(i)) = exp( -T*.5*(KLdivergence(H1,H2) + KLdivergence(H2,H1)));
    KLIST(i) = KL(r(i),c(i));
    
    if KL(r(i),c(i)) == 0
        disp('KL is zero for some reason!?');
        keyboard;
    end
    if isnan(KL(r(i),c(i)))
        disp('KL is NaN for some reason!?');
        keyboard;
    end
    
    if mod(i,1000) == 0
        disp([' KL progress (' num2str(i) '/' num2str(length(r)) ')']);
    end
    
end

% make KL symmetric
KL = max(KL,KL');


%% KL divergence for two pre-normalized 1-D histograms
function k = KLdivergence(H1n,H2n)

temp = H1n.*log(H1n./H2n);
temp(isnan(temp)) = 0;
k = sum(temp);

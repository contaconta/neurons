function [ASSIGN P] = assignFilaments(L, FRANGI, indList, priorList)


% parameters
A = -1.5524;
B = -31.1269;
BORDER = 10;
NEURON_PROB_THRESH = 0.000001; %0.000001;  %.001

% determine the # of somas
nSomas = numel(indList);


% compute frangi probability
%P = 1./(1+exp(A*log(FRANGI)+B));
P = 0.001 + .998./(1+exp(A*log(FRANGI)+B));

% make image border mask
M = zeros(size(FRANGI));
M = M > Inf;
M(1:end,1:BORDER) = 1; 
M(1:end,end-BORDER+1:end) = 1;
M(1:BORDER,1:end) = 1; 
M(end-BORDER+1:end,1:end) = 1;


% set image border probabilities to zero
P(M) = 0;
    
if nSomas ~= 0
    Iret = zeros(size(L,1), size(L,2), nSomas);
    for i = 1:nSomas
        SomaInd = indList(i);

        SomaMask = double(L > 0);
        SomaMask(L == SomaInd) = 2;
    %     SomaMask = (L == SomaInd);
    %     SomaMask = 255*SomaMask;

        IretTmp = ComputeProbabilityMapShortestPath(P, SomaMask);
        Iret(:,:,i) = IretTmp * priorList(i);
    end


    Pneuron = sum(Iret,3);  % ./ nSomas;
    idx = find(Pneuron >= NEURON_PROB_THRESH);
    %idx = find(P >= 0.7);
    FGMASK = zeros(size(P));
    FGMASK(idx) = 1;

    P_T = Pneuron;
    P_N = zeros(size(Iret));

    for i = 1:nSomas
        P_B = zeros(size(P));
        R = Iret(:,:,i)./P_T;
        P_B(idx) = R(idx);
        P_N(:,:,i) = P_B;
    end

    [Max, MaxInd] = max(P_N, [], 3);  %#ok<ASGLU>
    clear Max;
    MaxInd = MaxInd.*FGMASK;
    ASSIGN = MaxInd.*FGMASK;

    for i = 1:nSomas
        ASSIGN(MaxInd == i) = indList(i);
    end

else
    ASSIGN = zeros(size(L));
end
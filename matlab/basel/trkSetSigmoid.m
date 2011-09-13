function [Abest Bbest] = trkSetSigmoid(SL, f, G, G_STD, G_MAX, indList)



nmax = 400;

POS = [];
NEG = [];

% randomly select some data points
for t = 1:length(f)
    
    L = SL{t};
    
    nmax = min(nmax, sum(sum(L > 0)));
    
    posinds = find(L > 0);
    
    if nmax > 0
    
    
        pospts = f{t}(randsample(posinds,nmax));
        
        % mask of bg points
        %fmedian = median(f{t}(:));
        M = (G{t} > 2.5*(G_STD/G_MAX)) & (L == 0);
        %M = (f{t} > .5*fmedian) & (f{t} < 1.25*fmedian);
        neginds = find(M == 1);
        negpts = f{t}(randsample(neginds, nmax));
        
        POS = [POS; pospts];
        NEG = [NEG; negpts];
        
    end    
end


%keyboard;

best = 0; Abest = -3.5; Bbest = -50;

for A = -3.5:.05:-.75
    for B = -50:-5

        Ppos = 0.001 + .998./(1+exp(A*log(POS)+B));
        Pneg = 1 - (0.001 + .998./(1+exp(A*log(NEG)+B)));
        
        score = sum(Ppos) + sum(Pneg);
        
        if score > best
            Abest = A;
            Bbest = B;
            best = score;
        end
    end
end

% keyboard;
% 
% [Abest Bbest]
% P = 0.001 + .998./(1+exp(Abest*log(f{97})+Bbest));
% figure; imagesc(P);
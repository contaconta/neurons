function T = trkGreedyConnect2(W,A,D,W_THRESH)



min_W = 0;

T = zeros(size(A));

Ws = W + W';

idxWisNot0 = find(max(Ws)~=0);

reducedToComplete = idxWisNot0;
completeToReduced = zeros(size(W,1),1);
nColumn = 1;
for i = 1:size(W,1)
   if find(idxWisNot0 == i)
       completeToReduced(i) = nColumn;
       nColumn = nColumn + 1;
   end
end

Wr = Ws(idxWisNot0,idxWisNot0);
clear Ws;

Wr = tril(Wr);

Wr(Wr == 0) = Inf;

%keyboard;

while min_W < W_THRESH
    
    % find the minimum weight
    [min_W, min_ind] = min(Wr(:));
    if isinf(min_W)
        continue;
    end
    [rr,cr] = ind2sub(size(Wr), min_ind);
    r = reducedToComplete(rr);
    c = reducedToComplete(cr);
    
    % assign the connection in the tracking solution
    T(r,c) = 1;
    A(r,c) = 0;
    Wr(completeToReduced(r),completeToReduced(c)) = Inf;
    
    % deteriming which is the future, which is the past
    rt = D(r).Time;
    ct = D(c).Time;
    

    
    if rt == ct
        error('ERROR: equal time edges!');
    end
    
    if rt < ct
        past = r;  pt = rt;
        fut = c;   ft = ct;
    else
        past = c;  pt = ct;
        fut = r;   ft = rt;
    end
    
    % remove all future connections for PAST
    edgesr = find(A(past,:) == 1);
    edgesc = find(A(:,past) == 1);
    
    for i = 1:length(edgesr)
        if D(edgesr(i)).Time > pt;
           A(past, edgesr(i)) = 0;
              Wr(completeToReduced(past), completeToReduced(edgesr(i))) = Inf;

        end
    end
    for i = 1:length(edgesc)
        if D(edgesc(i)).Time > pt;
           A(edgesc(i), past) = 0;
              Wr(completeToReduced(edgesc(i)), completeToReduced(past)) = Inf;
         
        end
    end
    
    % remove all past connections for FUT
    edgesr = find(A(fut,:) == 1);
    edgesc = find(A(:,fut) == 1);
    for i = 1:length(edgesr)
        if D(edgesr(i)).Time < ft;
           A(fut, edgesr(i)) = 0;
              Wr(completeToReduced(fut), completeToReduced(edgesr(i))) = Inf;
         
        end
    end
    for i = 1:length(edgesc)
        if D(edgesc(i)).Time < ft;
           A(edgesc(i), fut) = 0;
              Wr(completeToReduced(edgesc(i)), completeToReduced(fut)) = Inf;
        end
    end
    %keyboard;
    
end

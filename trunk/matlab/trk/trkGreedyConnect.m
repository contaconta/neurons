function T = trkGreedyConnect(W,A,D,W_THRESH)

W(W == 0) = Inf;


min_W = 0;

T = zeros(size(A));

while min_W < W_THRESH
    
    % find the minimum weight
    min_W = min(min(W));
    if isinf(min_W)
        continue;
    end
    min_ind = find(W == min_W);
    [r,c] = ind2sub(size(A), min_ind);
    
%     if (r == 58) || (c == 58)
%         keyboard;
%     end
    
    % assign the connection in the tracking solution
    T(r,c) = 1;
    A(r,c) = 0;
    W(r,c) = Inf;
    
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
           W(past, edgesr(i)) = Inf;
        end
    end
    for i = 1:length(edgesc)
        if D(edgesc(i)).Time > pt;
           A(edgesc(i), past) = 0;
           W(edgesc(i), past) = Inf;
        end
    end
    
    % remove all past connections for FUT
    edgesr = find(A(fut,:) == 1);
    edgesc = find(A(:,fut) == 1);
    for i = 1:length(edgesr)
        if D(edgesr(i)).Time < ft;
           A(fut, edgesr(i)) = 0;
           W(fut, edgesr(i)) = Inf;
        end
    end
    for i = 1:length(edgesc)
        if D(edgesc(i)).Time < ft;
           A(edgesc(i), fut) = 0;
           W(edgesc(i), fut) = Inf;
        end
    end
    %keyboard;
    
end

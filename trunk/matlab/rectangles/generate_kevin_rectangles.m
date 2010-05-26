function [INDS, POLS] = generate_kevin_rectangles(IMSIZE, RANK, CONNECTEDNESS)


% keep a record of where we've placed rectangles before
Q = zeros(IMSIZE);
INDS = cell(1,RANK); POLS = zeros(1,RANK);

POS_AREA = 1;
NEG_AREA = 1;

%CONNECTEDNESS = .9;

for i = 1:RANK
    
    % attempt to generate a valid rectangle
    valid = []; iters = 0;
    while isempty(valid) 
        [valid w h N] = generate_valid_rectangle(IMSIZE, Q);
        iters = iters + 1;
        if iters > 100
            disp('   could not find a valid place for the rectangle! Exiting.');
            return
        end
    end
    
    % set the neighbor weights
    %N = N + 1;
    N(N==1) =  CONNECTEDNESS / sum(N==1);
    N(N==0) = (1-CONNECTEDNESS) / sum(N==0);
    
    % sample one of the safe locations
    ind = randsample(size(valid,1),1,true,N);
    r1 = valid(ind,1);
    c1 = valid(ind,2);
    
    %keyboard;
    
    % place the rectangle
    Q(r1:r1+h-1,c1:c1+w-1) = 1;
    %Q(r1:r1+h-1,c1:c1+w-1) = i;
    %figure(3);
    %imagesc(Q);
    
    R = [r1 r1 r1+h r1+h];
    C = [c1 c1+w c1 c1+w];
    IND = sub2ind(IMSIZE+[1 1], R, C);
    INDS{i} = IND;
    
    % sample the polarity so that it will be approx equal
    polset = [-1 1];
    POLS(i) = randsample(polset,1, true, [POS_AREA NEG_AREA]);
    if POLS(i) == 1
        POS_AREA = POS_AREA + w*h;
    else
        NEG_AREA = NEG_AREA + w*h;
    end
    
    
    
end






function [valid w h N] = generate_valid_rectangle(IMSIZE, Q)

% SIGMAW = (IMSIZE(2)-1)/3;
% SIGMAH = (IMSIZE(1)-1)/3;

% SIGMAW = (IMSIZE(2)-1)/5;
% SIGMAH = (IMSIZE(1)-1)/5;

SIGMAW = (IMSIZE(2)-1)/7;
SIGMAH = (IMSIZE(1)-1)/7;

% SIGMAW = (IMSIZE(2)-1)/7;
% SIGMAH = (IMSIZE(1)-1)/7;

% SIGMAW = (IMSIZE(2)-1);
% SIGMAH = (IMSIZE(1)-1);

% sample a rectangle size
w = [];
while isempty(w)
    w_ = 1 + round(abs( SIGMAW*randn(1,1)));
    if w_ <= IMSIZE(2)
        w = w_;
    end
end

h = [];
while isempty(h)
    h_ = 1 + round(abs( SIGMAH*randn(1,1)));
    if h_ <= IMSIZE(1)
        h = h_;
    end
end


% is it connected?

% scan to find safe places to place it
valid = zeros(numel(Q), 2);  q = 0; N = zeros(numel(Q),1);
for r = 1:IMSIZE(1)-h+1
    for c = 1:IMSIZE(2)-w+1
        r2 = r + h - 1;
        c2 = c + w - 1;

        if sum(Q(r:r2,c:c2)) == 0
            q = q + 1;
            valid(q,:) =  [r c];
        
        
            % check for neighbors
            if r > 1
                if sum(Q(r-1, c:c2)) ~=0
                    N(q) = 1;
                    continue;
                end
            end
            if (c > 1) %&& (N(q) == 0)
                if sum(Q(r:r2,c-1)) ~=0
                    N(q) = 1;
                    continue;
                end
            end        
            if (r2 < IMSIZE(1)) %&& (N(q) == 0)
                if sum(Q(r2+1,c:c2)) ~= 0
                    N(q) = 1;
                    continue;
                end
            end
            if (c2 < IMSIZE(2)) %&& (N(q) == 0)
                if sum(Q(r:r2,c2+1)) ~= 0
                    N(q) = 1;
                    continue;
                end
            end
        end
    end
end

N = N( valid(:,1) ~= 0, :);
valid = valid( valid(:,1) ~= 0,:);

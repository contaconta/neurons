function [RECTS, COLS] = generate_asymmetric_rectangles(N, IMSIZE, WH1, CONNECTEDNESS)

% WH1 = the max size of the small rectangle

    RANK = 2;
    RECTS = cell(N,1);
    COLS = cell(N,1);
    MinAreaRatio = 3;

for i = 1:N
    %% generate the small rectangle

    rects = cell(1,RANK); cols = zeros(1,RANK);

    % record of previously placed rectangle(s)
    Q = zeros(IMSIZE);

    % place the first rectangle placement
    r = randint([1 24],1);
    c = randint([1 24],1);

    WMAX = IMSIZE(2) - c + 1;
    HMAX = IMSIZE(1) - r + 1;

    w = randsample(1:WH1,1,true,1./(1:WH1));  if c+w > IMSIZE(2); w = WMAX; end;
    h = randsample(1:WH1,1,true,1./(1:WH1));  if r+h > IMSIZE(1); h = HMAX; end;
    Ws = w; Hs = h;
    %disp([ num2str(w) ' ' num2str(h)]);

    R = [r r r+h r+h];
    C = [c c+w c c+w];
    rect = sub2ind(IMSIZE+[1 1], R, C);
    
    Q(r:r+h-1,c:c+w-1) = 1;

    rects{1} = rect;
    polset = [-1 1];
    cols(1) = randsample(polset,1);


    %% generate the big rectangle
    
    % sample if it is connected or not
    %CONNECT = rand(1) <= CONNECTEDNESS;
    
    valid = []; iters = 0;
    while isempty(valid) 
        [valid w h N] = generate_valid_rectangle(IMSIZE, Ws, Hs, Q, MinAreaRatio);
        iters = iters + 1;
        if iters > 100
            disp('   could not find a valid place for the rectangle! Exiting.');
            return
        end
    end
    
    % set the neighbor weights
    N(N==1) =  CONNECTEDNESS / sum(N==1);
    N(N==0) = (1-CONNECTEDNESS) / sum(N==0);
    
    %keyboard;
    
    % sample one of the safe locations
    ind = randsample(size(valid,1),1,true,N);
    r1 = valid(ind,1);
    c1 = valid(ind,2);
    
    R = [r1 r1 r1+h r1+h];
    C = [c1 c1+w c1 c1+w];
    rect = sub2ind(IMSIZE+[1 1], R, C);
    
    rects{2} = rect;
    cols(2) = -cols(1);
    
    
    RECTS{i} = rects;
    COLS{i} = cols;

end






function [valid w h N] = generate_valid_rectangle(IMSIZE, Ws, Hs, Q, MinAreaRatio)


WHmin = MinAreaRatio*min(Ws,Hs);
%Amin = 3*Ws*Hs;

validW = WHmin:IMSIZE(2)-1;
validH = WHmin:IMSIZE(1)-1;

w = randsample(validW,1,true,1./validW); 
h = randsample(validH,1,true,1./validH); 


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
                    %keyboard;
                    continue;
                end
            end
            if (c > 1) %&& (N(q) == 0)
                if sum(Q(r:r2,c-1)) ~=0
                    N(q) = 1;
                    %keyboard;
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




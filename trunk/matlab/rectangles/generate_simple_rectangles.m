function [rects, cols] = generate_simple_rectangles(N, IMSIZE, RANK)

rects = cell(N,1);
cols = cell(N,1);

for i = 1:N
    RANKi = randint([2 RANK],1);
    [rect, col] = generate_simple_rectangle(IMSIZE, RANKi);
    rects{i} = rect;
    cols{i} = col;
end


function [rects cols] = generate_simple_rectangle(IMSIZE, RANK)

rects = cell(1,RANK); cols = zeros(1,RANK);

for i = 1:RANK
    
    % sample the r and c coordinates
    r = randint([1 24],1);
    c = randint([1 24],1);
    
    % determine valid W and H
    WMAX = IMSIZE(2) - c + 1;
    HMAX = IMSIZE(1) - r + 1;
    
    % sample the w and h
    w = randint([1 WMAX],1);
    h = randint([1 HMAX],1);
 
    R = [r r r+h r+h];
    C = [c c+w c c+w];
    rect = sub2ind(IMSIZE+[1 1], R, C);
    
    rects{i} = rect;
    
    polset = [-1 1];
    %cols(i) = randsample(polset,1, true, [POS_AREA NEG_AREA]);
    cols(i) = randsample(polset,1);
end


% ensure that there is at least 1 rectangle of each color
while length(unique(cols)) == 1
    cols = randsample(polset, RANK, true);
end
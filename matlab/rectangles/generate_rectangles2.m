function [ROWS, COLS, INDS, POLS] = generate_rectangles2(N,IMSIZE, RANK, CONNECTEDNESS)
%
%
%
%
%
%
%

%CONNECTEDNESS = 0.9;
ROWS = [];
COLS = [];
INDS = cell(N,1);
POLS = cell(N,1);

for n = 1:N

    rnk = randi([2 RANK]);
    
    [rect, pols] = generate_kevin_rectangles(IMSIZE, rnk, CONNECTEDNESS); 
    
    INDS{n} = rect;
    POLS{n} = pols;
    
end
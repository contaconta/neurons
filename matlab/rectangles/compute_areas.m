function A = compute_areas(IMSIZE, rects)

IMSIZE1 = IMSIZE+[1 1];
A = cell(size(rects));

for f = 1:length(rects)
    
    rectsf = rects{f};
    
    fA = zeros(1,length(rectsf));
    
    for j = 1:length(rectsf)
        
        rect = rectsf{j};
    
        [r c] = ind2sub(IMSIZE1, rect);
        
        fA(j) = (r(4)-r(1))  * (c(4)-c(1));
    end
    
    A{f} = fA;
    
end
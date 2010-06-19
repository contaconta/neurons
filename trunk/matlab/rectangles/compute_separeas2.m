function A = compute_separeas2(IMSIZE,  rects)

IMSIZE1 = IMSIZE+[1 1];
A = cell(size(rects));

for f = 1:length(rects)
    
    rectsf = rects{f};

    separeas = zeros(size(rectsf));
    
    for j = 1:length(rectsf)
        
        rect = rectsf{j};
    
        [r c] = ind2sub(IMSIZE1, rect);
        
        thisarea = (r(4)-r(1))  * (c(4)-c(1));
        
        separeas(j) = thisarea;
       
    end
    
    A{f} = separeas;
end

% white areas is firs
function A = compute_nonorm_areas(rects)

A = cell(size(rects));

for f = 1:length(rects)
    
 
    
    A{f} = [0 0];   % if they are equal area, set area to 0 so we skip the normalization!
   
    
end

% white areas is first, black is 2nd
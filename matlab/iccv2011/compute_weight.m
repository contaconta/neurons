function W = compute_weight(R,C,P)

W = cell(size(R));

for i = 1:length(R)
    
    Ri = R{i};
    
    area = zeros(size(Ri));
    
    for k = 1:length(Ri)
    
        Rik = R{i}{k};
        Cik = C{i}{k};
        
        area(k) = (Rik(2) - Rik(1) + 1) * (Cik(2) - Cik(1) + 1);
        
    end
    
    whiteinds = find(P{i} == 1);
    blackinds = find(P{i} == -1);

    whitearea = sum(area( whiteinds ));
    blackarea = sum(area( blackinds ));
    
   	% whiteregion weights should always sum to 1
    area(whiteinds) = area(whiteinds) ./ whitearea;
    
    % blackregion weights should also sum to 1
    %blackwhiteratio = whitearea/blackarea;
    %area(blackinds) = -1* blackwhiteratio .* (area(blackinds) ./ blackarea);
    area(blackinds) = -1 * (area(blackinds) ./ blackarea);
    
    %area(whiteinds) = area(whiteinds) ./ whitearea;
    %area(blackinds) = -1 * (area(blackinds) ./ blackarea);
    
    W{i} = area;
    
end
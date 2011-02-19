function h = drawLabels(L)


count = 1;

for r = .5:1:size(L,1)-.5
    for c = .5:1:size(L,2)-.5
        
        rI = r+.5;
        cI = c+.5;
        
        % test right
        if (cI ~= size(L,2))
            if L(rI,cI) ~= L(rI,cI+1)
                h(count) = plot([c+1 c+1], [r r+1],'r-');
                count = count + 1;
            end
        end
        
        % test down
        if (rI ~= size(L,1))
            if L(rI,cI) ~= L(rI+1,cI)
                h(count) = plot([c c+1],[r+1 r+1], 'r-');
                count = count + 1;
            end
        end
        
    end
end
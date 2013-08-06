function S = convert_GT_to_S(GT)


% clockwise vertices, start point = end point
Tmax = size(GT{1},1);
num_tracks_GT = numel(GT);
fprintf('converting GT to S');
for i = 1:num_tracks_GT
    fprintf('.');
    for t = 1:Tmax
        cmin = GT{i}(t,1);
        rmin = GT{i}(t,2);
        width = GT{i}(t,3);
        height = GT{i}(t,4);
        
        if cmin == 0
            S(i).P(t).x = [];
            S(i).P(t).y = [];
        else
            S(i).P(t).x = [cmin cmin        cmin+width  cmin+width cmin];
            S(i).P(t).y = [rmin rmin+height rmin+height rmin       rmin];
        end
    end
end
fprintf('\n');
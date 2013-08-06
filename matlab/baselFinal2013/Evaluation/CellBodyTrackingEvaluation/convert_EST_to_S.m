function S = convert_EST_to_S(EST)

C           = 696;
R           = 520;

% clockwise vertices, start point = end point
Tmax = 97;
num_tracks = numel(EST.TrackedCells);

for i = 1:num_tracks
    
    tList = [EST.TrackedCells(i).TimeStep(:).Time];
    for t = 1:Tmax
        
        B = zeros(R,C);
        
        ind = find(tList == t, 1);
        
        if isempty(ind)
            S(i).P(t).x = [];
            S(i).P(t).y = [];
        else
            est = EST.TrackedCells(i).TimeStep(ind);
            pixlist = est.SomaPixelIdxList;
            B(pixlist) = 1;
            
            BW = bwboundaries(B, 8, 'noholes');  % 4 or 8 connectivity?
            
            if numel(BW) > 1
                keyboard;
            else                
                x1 = BW{1}(:,2);
                y1 = BW{1}(:,1);
                [x2, y2] = poly2cw(x1, y1);
                S(i).P(t).x = x2;
                S(i).P(t).y = y2;
            end
        end
    end
end



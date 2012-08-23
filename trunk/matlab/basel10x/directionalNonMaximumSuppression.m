function DNMS = directionalNonMaximumSuppression(I, NUM_DIRECTIONS)




WINDOW_LENGTH  = 13;
Percent = .01;
SMALL_OBJECT_SIZE = 20;
LEFT_RIGHT_PERCENT = .3;


I = double(I);
DNMS = zeros(size(I));

dNMS = cell(1,NUM_DIRECTIONS);



for i = 1:NUM_DIRECTIONS
    angle = ((i-1)/(NUM_DIRECTIONS)) * 180;
   
    [R C] = lineBreshenham(WINDOW_LENGTH, angle);
    maskSize = [max(R) max(C)];
    
    M = makeMask(R,C,maskSize);
    DirectionalMax = ordfilt2(I, WINDOW_LENGTH, M);

    
    edgeSize = max(1, round(LEFT_RIGHT_PERCENT*WINDOW_LENGTH));
    
%     L = makeMask(R(1:edgeSize),C(1:edgeSize),maskSize);
%     R = makeMask(R(end-edgeSize+1:end),C(end-edgeSize+1:end),maskSize);
%     AvgLeft = ordfilt2(I,edgeSize,L);
%     AvgRight = ordfilt2(I,edgeSize,R);
    
    L = makeMask(R(1:edgeSize),C(1:edgeSize),maskSize)/edgeSize;
    R = makeMask(R(end-edgeSize+1:end),C(end-edgeSize+1:end),maskSize)/edgeSize;
    AvgLeft = imfilter(I,L);
    AvgRight = imfilter(I,R);
    
%     dNMS{i} = (I == DirectionalMax) .* (I - AvgLeft >= Idiff) .* (I - AvgRight >= Idiff);
    dNMS{i} = (I == DirectionalMax) .* (I - AvgLeft >= Percent.*I) .* (I - AvgRight >= Percent.*I);
end


% combine each direction into a single image
for i = 1:NUM_DIRECTIONS
    DNMS = DNMS + dNMS{i};
end
DNMS = DNMS > 0;


% remove small objects
props  = regionprops(DNMS > 0, 'Area', 'PixelIdxList');
for i = 1:numel(props)
    if props(i).Area < SMALL_OBJECT_SIZE
        DNMS(props(i).PixelIdxList) = 0;
    end
end


% TODO: LINKING BROKEN 

% figure(9);
% imagesc(DNMS);







function [R C] = lineBreshenham(n, angle)

slope = abs(tand(angle));
invslope = 1/slope;
sr = sign(sind(angle));                             % row sign
sc = sign(cosd(angle));                             % col sign
steep = abs(slope) > 1;                             % is it a 'steep' line?
if steep; e = invslope; else e = slope; end         % set initial error

R = zeros(n,1);
C = zeros(n,1);

r = 0;
c = 0;

R(1) = r;
C(1) = c;

for i = 2:n
    if steep
        % row-dominant region; increase r
        r = r + sr;
        e = e + invslope;
        
        if abs(e >= 0.5)
            c = c + sc;
            e = e - 1;
        end
        R(i) = r;
        C(i) = c;
        
    else
        % col-dominant region; increase c
        c = c + sc;
        e = e + slope;
        
        if abs(e >= 0.5)
            r = r + sr;
            e = e - 1;
        end
        
        R(i) = r;
        C(i) = c;
    end
    
    
end

rmin = min(R);
cmin = min(C);
R = R - rmin + 1;
C = C - cmin + 1;




function M = makeMask(R,C,MaskSize)

M = zeros(MaskSize);
for j = 1:numel(R)
    M(R(j),C(j)) = 1;
end



    
    
    
    
    % % frangi parameters
% opt.FrangiScaleRange = [1 2];
% opt.FrangiScaleRatio = 1;
% opt.FrangiBetaOne = .5;
% opt.FrangiBetaTwo = 15;
% opt.BlackWhite = false;
% opt.verbose = false;

% I = imread('/home/ksmith/code/neurons/matlab/basel/testImages/Green/im0001.TIF');
% F = FrangiFilter2D(mat2gray(double(I)), opt);
% figure; imagesc(I);
% figure; imagesc(F); 


 
% Idiff = 0;
% NUM_DIRECTIONS = 4;  % 8

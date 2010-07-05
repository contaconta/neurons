function DS = denseScaleEstimate(I, CONST)

% parameters
if ~exist('CONST', 'var')
    CONST = 100;        % constant gradient sum in an area around pixel
end
PADR = 100;          % image padding for rows    
PADC = 100;          % image padding for columns


% pad the initial image using symmetric values from the original
Ipad = padarray(I, [PADR PADC], 'symmetric');

Ipad = imgaussian(Ipad, 1);

% compute the gradient in x and y
[gx gy]= gradient(double(Ipad));

% compute the gradient norm
G(:,:,1) = gx;  G(:,:,2) = gy;
GN = sqrt(sum((G.^2),3));


% compute the integral image of the gradient norm
intimg = integralImage(GN);
SCALE = zeros(size(Ipad));


%% scan the image (exclude the padding). At each pixel, estimate the scale
for r = PADR+1:size(Ipad,1)-PADR
    for c = PADC+1:size(Ipad,2)-PADC
        
        
        s = findscale(intimg,r,c, CONST, PADR, PADC);
        
        
        SCALE(r,c) = s;
        
    end
end


%% remove the padding from the 
DS = SCALE(PADR+1:size(SCALE,1)-PADR, PADC+1:size(SCALE,2)-PADC);






function s = findscale(I,r,c, CONST, PADR, PADC)

s = 0;
%keyboard;
while I(r+s,c+s) + I(r-s,c-s) - I(r+s,c-s) - I(r-s,c+s) < CONST
    s = s + 1;
    
    if (s > PADR) || (s > PADC)
        return
    end
end
%keyboard;


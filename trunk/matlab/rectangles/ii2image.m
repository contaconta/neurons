function I = ii2image(ii, IMSIZE, type)
% transforms an integral image back into an image
%



if strcmp(type, 'outer')
    
    I = zeros(IMSIZE, 'single');

    ii = reshape(ii, IMSIZE + [1 1 ]);
    
    for r = 1:IMSIZE(1)
        for c = 1:IMSIZE(2)
            
            I(r,c) = ii(r+1,c+1) - ii(r,c+1) + ii(r,c) - ii(r+1,c);
            
        end
    end
end

I = uint8(I);
folder = '/home/ksmith/data/faces/EPFL-CVLAB_faceDB/train/pos/';

d = dir([folder '*.png']);

%f = [3 0 0.5 16 0 18 7 0 -1 16 8 18 15 0 0.5 16 16 18 23];
%f = [3 0 0.5 8 12 10 14 0 -1 11 12 13 14 0 0.5 14 12 16 14];
f = [3 0 0.5 9 0 10 7 0 -1 11 0 12 7 0 0.5 13 0 14 7];

for i = 1:10
    
    I{i} = imread([folder d(i).name]);
    IMSIZE = size(I{i});
    
    
    BW = rectRender(f, IMSIZE);
    
    totalsum = 0;
    
    p = 2;
    RANK = f(1);
    for k = 1:RANK
        w = f(p+1);
        x0 = f(p+2)+1;
        y0 = f(p+3)+1;
        x1 = f(p+4)+1;
        y1 = f(p+5)+1;
        
        weightedsum = w*sum(sum(I{i}(y0:y1,x0:x1)));

        disp(['sum of region ' num2str(k) ': ' num2str(weightedsum)]);
        totalsum = totalsum + weightedsum;
        
        p = p + 6;
    end
        
        
    disp(totalsum);
    
 %   whiteregion = sum(I{i}(BW == 1));
 %   blackregion = sum(I{i}(BW == -1));
    
    %disp(['r = W - B   ' num2str(whiteregion-blackregion) ' = ' num2str(whiteregion) ' - ' num2str(blackregion)]);
    
    Idisp = I{i};
    Idisp(BW == 1) = 255;
    Idisp(BW == -1) = 0;
    imshow(Idisp);
    pause;
end
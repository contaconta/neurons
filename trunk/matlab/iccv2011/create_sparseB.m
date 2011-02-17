
gPerLine = 5;
sigmas = 1:8;
IMSIZE = [24 24];
reflectProb = .333;

Nfeatures = 500000;

filename = ['sparseB' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


fid = fopen(filename, 'w');
fprintf(fid, '%d\n', Nfeatures);

disp(['...writing to ' filename]);


for i = 1:Nfeatures
    
    [x y w s] = gaussianRandomShape(IMSIZE, sigmas, gPerLine, 1, 1, reflectProb);
    
    %R = reconstruction(IMSIZE, x, y, w, s);
    %imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]); colormap gray;
    %drawnow;
    
    n = length(x);
    xc = mean(x);
    yc = mean(y);
    
    str = [num2str(n) sprintf(' %f %f', xc, yc)];
    
    for k = 1:n        
        str = [str sprintf(' %f %f %d %d',  w(k), s(k), x(k), y(k) )];
    end
    
    fprintf(fid, [str '\n']);
    
    if mod(i, 2000) ==  0
        disp(['...generated ' num2str(i)]);
    end
end


fclose(fid);
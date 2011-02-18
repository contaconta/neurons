warning('off','MATLAB:polyfit:PolyNotUnique');
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale');
sigmas = [.5 1:8];
IMSIZE = [24 24];
reflectProb = .85;

Nfeatures = 500000;

filename = ['sparseE' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


fid = fopen(filename, 'w');
fprintf(fid, '%d\n', Nfeatures);

disp(['...writing to ' filename]);


for i = 1:Nfeatures
    
    %K = randi([1 10]);
    K = abs(round(2.5*randn(1)))+1;
    NPOS = max(1,round(randn(1)));
    NNEG = max(1,round(randn(1)));
    SIGO = pi/randi([1 6]);
    %[K rad2deg(SIGO) NPOS NNEG]
    
    [x y w s] = gaussianRandomShape(IMSIZE, sigmas,K,NPOS,NNEG, reflectProb, 0, SIGO);
    
    n = length(x);
    mux = mean(x);
    muy = mean(y);
    xc = (1/n)* sum( (((x-mux) ./ s)+mux) );
    yc = (1/n)* sum( (((y-muy) ./ s)+muy) );
    

    
    str = [num2str(n) sprintf(' %f %f', xc, yc)];
    
    for k = 1:n        
        str = [str sprintf(' %f %f %d %d',  w(k), s(k), x(k), y(k) )]; %#ok<AGROW>
    end
    
    fprintf(fid, [str '\n']);
    
    if mod(i, 2000) ==  0
        disp(['...generated ' num2str(i) ' (' num2str(numel(x)) ') samples: ' str]);
        R = reconstruction(IMSIZE, x, y, w, s);
        imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]); colormap gray; hold on;
        plot(xc+1,yc+1, 'mo'); hold off;
        drawnow;
        %pause;
    end
end


fclose(fid);
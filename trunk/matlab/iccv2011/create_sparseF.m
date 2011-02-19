warning('off','MATLAB:polyfit:PolyNotUnique');
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale');
sigmas = [.5 1:8];
IMSIZE = [24 24];
reflectProb = .75;
crossProb = .25;

Nfeatures = 500000;

filename = ['pathF' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


fid = fopen(filename, 'w');
fprintf(fid, '%d\n', Nfeatures);

disp(['...writing to ' filename]);


for i = 1:Nfeatures
    
    %K = abs(round(3*randn(1)))+1;
    K = randsample(1:8,1,true,8:1);
    N = randsample([2 3], 1, true, [2/3 1/3]); 
    SIGO = pi/randsample(2:2:12,1);
    %[K rad2deg(SIGO) N]
    
    typeweights = [1 1 1 .25];
    
    [x y w s] = gaussianRandomShape(IMSIZE,sigmas,K,N,typeweights,SIGO);
    
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
    
    if mod(i, 1) ==  0
        disp(['...generated ' num2str(i) ' (' num2str(numel(x)) ') samples: ' str]);
        R = reconstruction(IMSIZE, x, y, w, s);
        imagesc(R,[-max(abs(R(:))) max(abs(R(:)))]); colormap gray; hold on;
        plot(x(w > 0)+1, y(w > 0)+1, 'rs');
        plot(x(w < 0)+1, y(w < 0)+1, 'g.'); 
        plot(xc+1,yc+1, 'mo'); hold off;
        drawnow;
        pause;
    end
end


fclose(fid);

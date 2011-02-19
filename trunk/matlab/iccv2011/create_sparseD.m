warning('off','MATLAB:polyfit:PolyNotUnique');
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale');
sigmas = [.5 1:8];
IMSIZE = [24 24];
reflectProb = .75;
crossProb = .25;

Nfeatures = 500000;

filename = ['pathD' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];


fid = fopen(filename, 'w');
fprintf(fid, '%d\n', Nfeatures);

disp(['...writing to ' filename]);


for i = 1:Nfeatures
    
    %K = randi([1 10]);
    K = abs(round(3*randn(1)))+1;
    %K = randi([1 10],1);
    N = randsample([2 3], 1, true, [2/3 1/3]);
    SIGO = pi/randi([1 6]);
    %[K rad2deg(SIGO) NPOS NNEG]
    
    [x y w s] = gaussianRandomShape(IMSIZE, sigmas,K,N, reflectProb, crossProb, SIGO);
    
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

%cmd = 'scp sparseD24x24.list cvlabpc11:/home/ksmith/neurons/matlab/iccv2011/';
%system(cmd);
filename = 'violajones24x24.list';


IMSIZE = [24 24]; BLANK = zeros(IMSIZE);

fid=fopen(filename);
tline = fgetl(fid);

nfeatures = str2double(tline);

outname = ['randomdirac_violajones' num2str(IMSIZE(1)) 'x' num2str(IMSIZE(2)) '.list'];

fid2 = fopen(outname, 'w');
fprintf(fid2, '%d\n', nfeatures);

disp(['...writing to ' outname]);

for i = 1:nfeatures
    tline = fgetl(fid);

    f = str2num(tline); %#ok<ST2NM>
    
    
    [X Y W S XC YC] = randomDiracSampling(f);
    
    n = length(X);
    str = [num2str(n) sprintf(' %f %f', XC, YC)];
    for k = 1:n        
        str = [str sprintf(' %f %f %d %d',  W(k), S(k), X(k), Y(k) )]; %#ok<*AGROW>
    end
    
    if mod(i, 2000) == 0
        disp(['feature ' num2str(i) ' (' num2str(numel(X)) ') samples: ' str]);
        BW = rectRender(f, IMSIZE, BLANK);
        clf; imagesc(BW);  colormap gray; hold on;
        plot(X(W > 0)+1, Y(W > 0)+1, 'rs');
        plot(X(W < 0)+1, Y(W < 0)+1, 'gs'); 
        plot(XC,YC, 'mo'); hold off;
        drawnow;
    end
    
    fprintf(fid2, [str '\n']);
    
    
end

fclose(fid);
fclose(fid2);

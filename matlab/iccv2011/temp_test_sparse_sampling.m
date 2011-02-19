filename = 'violajones24x24.list';



IMSIZE = [24 24]; BLANK = zeros(IMSIZE);

fid=fopen(filename);
tline = fgetl(fid);

nfeatures = str2double(tline);


for i = 1:nfeatures
    tline = fgetl(fid);

    f = str2num(tline); %#ok<ST2NM>
    
    
    [X Y W S] = randomDiracSampling(f);
    
    if mod(i, 2000) == 0
        disp(['feature ' num2str(i) ' (' num2str(numel(X)) ') samples: ' tline]);
        BW = rectRender(f, IMSIZE, BLANK);
        clf; imagesc(BW); colormap gray; hold on;
        plot(X(W > 0)+1, Y(W > 0)+1, 'rs');
        plot(X(W < 0)+1, Y(W < 0)+1, 'gs'); hold off;
        pause;
    end
    

end
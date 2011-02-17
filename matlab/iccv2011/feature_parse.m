function feature_parse(filename)

%filename = 'violajones24x24.list';



IMSIZE = [24 24];
BLANK = zeros(IMSIZE); figure(1); imagesc(BLANK); colormap gray;

fid=fopen(filename);
tline = fgetl(fid);

nfeatures = str2double(tline);

h = figure(1); colormap gray;
pos = get(h, 'Position');
set(h, 'Position', [pos(1), pos(2), 125 125]);


for i = 1:nfeatures
    tline = fgetl(fid);

    f = str2num(tline); %#ok<ST2NM>
    BW = rectRender(f, IMSIZE, BLANK);
    
    if f(2) == 0
        imagesc(BW); 
    else
        cla;
        imagesc(BW);
    end
    
 
    imagesc(BW);
    drawnow;
    
    
end

fclose(fid);
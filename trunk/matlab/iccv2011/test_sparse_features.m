%filename = 'strong.classifier.1500';
filename = 'strong.classifier.1500_0';

IMSIZE = [24 24];

fid=fopen(filename);
tline = fgetl(fid);

nfeatures = str2double(tline);

I = imread('/home/ksmith/data/faces/EPFL-CVLAB_faceDB/train/pos/face00001.png');
ALL = zeros([IMSIZE, nfeatures]);
S = [];

for i = 1:nfeatures
    
    tline = fgetl(fid);
    tline = strrep(tline, 't', ' ');
    
    f = str2num(tline); %#ok<ST2NM>
    
    f = [f(2) f(6:end-1)];
    pol = sign(f(end));
    
    [R s] = sparseRenderKarim(f, IMSIZE, pol);
    
    S = [S; s];
    
    %ALL(:,:,i) = abs(f(end))*R;
    ALL(:,:,i) = R;
    %pause;
    
end


mf = mean(ALL,3);
imagesc(mf);


figure; 
hist(S, numel(S));
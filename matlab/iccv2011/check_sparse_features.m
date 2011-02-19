clear;
folder = '/home/ksmith/data/faces/EPFL-CVLAB_faceDB/train/pos/';

d = dir([folder '*.png']);
filename = 'strong.classifier.1500_0';
IMSIZE = [24 24];
Sigmas = [.5 1:8];

% get the number of features
fid=fopen(filename);
tline = fgetl(fid);
nfeatures = str2double(tline);

% get the 1st feature
tline = fgetl(fid);
tline = strrep(tline, 't', ' ');
fstr = tline;
f = str2num(tline); %#ok<ST2NM>
xc = f(3);
yc = f(4);
f = [f(2) f(5:end-1)];
pol = sign(f(end));

figure(7);
[R s] = sparseRenderKarim(f, IMSIZE, pol,xc,yc);



for i = 1:10
    
    
    disp('---------------------------------');
    disp(['f = ' fstr]);
    disp(['image = ' folder d(i).name]);
    disp(' ');
    
    I{i} = imread([folder d(i).name]);
    IMSIZE = size(I{i});
    
    % convolve the gaussians
    for g = 1:length(Sigmas)
        G{g} = imgaussian(I{i}, Sigmas(g));
    end
    
    F = 0;
    Fquant = 0;
    
    RANK = f(1);
    x = zeros(RANK,1);
    y = zeros(RANK,1);
    w = zeros(RANK,1);
    s = zeros(RANK,1);

    p = 2;

    for k = 1:RANK
        w(k) = pol*f(p);
        s(k) = f(p+1);
        x(k) = f(p+2);
        y(k) = f(p+3);

        %[w(k) s(k) x(k) y(k)]

        p = p + 4;
        
        gind = find(Sigmas == s(k));
        
        F = F + w(k) * G{gind}(y(k)+1,x(k)+1);
        Fquant = Fquant + w(k) * round( G{gind}(y(k)+1,x(k)+1) );
    end
    
    [w s x y]
    [F Fquant]
    pause;
end

fclose(fid);
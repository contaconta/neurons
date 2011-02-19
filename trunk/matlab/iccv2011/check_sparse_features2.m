clear;
folder = '/home/ksmith/data/faces/EPFL-CVLAB_faceDB/train/pos/';

d = dir([folder '*.png']);
%filename = 'strong.classifier.1500_0';
filename = 'sparseViolaJonesK32-32_24x24.list';
IMSIZE = [24 24];
Sigmas = [.5 1:8];

% get the number of features
fid=fopen(filename);
tline = fgetl(fid);
nfeatures = str2double(tline);

% get the 1st feature
for k = 1:100
    tline = fgetl(fid);
    tline = strrep(tline, 't', ' ');
    fstr = tline;
    f = str2num(tline); %#ok<ST2NM>
    xc = f(2);
    yc = f(3);
    f = [f(1) f(4:end)];
    %pol = sign(f(end));
    pol = 1;
end
    


%keyboard;

for i = 1:100
    
    
    for k = 1:1000
        tline = fgetl(fid);
        tline = strrep(tline, 't', ' ');
        fstr = tline;
        f = str2num(tline); %#ok<ST2NM>
        xc = f(2);
        yc = f(3);
        f = [f(1) f(4:end)];
        pol = 1;
    end
    
    figure(7);
    [R s] = sparseRenderKarim(f, IMSIZE, pol,xc,yc);
    
    
    disp('---------------------------------');
    disp(['f = ' fstr]);
    disp(['image = ' folder d(i).name]);
    disp(' ');
    
    I{i} = imread([folder d(i).name]);
    IMSIZE = size(I{i});
    
    % convolve the gaussians
    for g = 1:length(Sigmas)
        G{g} = imgaussian2(I{i}, Sigmas(g));
        
%         kernelsize = [2*Sigmas(g)+1 2*Sigmas(g)+1];
%         if kernelsize(1) < 3
%             kernelsize(1) = 1;
%         end
%         if kernelsize(2) < 3
%             kernelsize(2) = 1;
%         end
%         h = fspecial('gaussian', kernelsize, Sigmas(g));
%         G{g} = imfilter(double(I{i}), h, 'replicate');
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
        
        Gval(k) = G{gind}(y(k)+1,x(k)+1);
        Imval(k) = I{i}(y(k)+1,x(k)+1);
        
        Fthis(k) = w(k) * G{gind}(y(k)+1,x(k)+1);
        F = F + Fthis(k);
        Fthisround(k) = w(k) * round( G{gind}(y(k)+1,x(k)+1));
        Fquant = Fquant + Fthisround(k);
        %keyboard;
    end
    
    %R1 = reconstruction(IMSIZE, x,y,w,s);
    R2 = reconstruction_coarse(IMSIZE,x,y,w,s);
    
    %figure(8); imagesc(R1); colormap gray;
    figure(9); imagesc(R2); colormap gray;
    
    %[w s x y ]
    %[Fthis(:) Fthisround(:)]
    %[Gval(:) round(Gval(:)) Imval(:)]
    [F Fquant]
    keyboard;
end

fclose(fid);
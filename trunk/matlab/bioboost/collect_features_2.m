function collect_features_2()

ann_folder = '/osshare/Work/Data/phase/annotations/';
img_folder = '/osshare/Work/Data/phase/images/';
load DATA_LOCS.mat

% add path to frangi filter
addpath('/osshare/Work/neurons/matlab/toolboxes/frangi_filter_version2a/');

d_img = dir([img_folder '*.png']);

PADSIZE = [10 10];  OFFSET = PADSIZE; %#ok<*NASGU>
WINDOW = [9 9]; patchsize = 2*WINDOW + [1 1];

Dlength = 0;
for i = 1:length(files);
    Dlength = Dlength + size(datapoints{i},1);
end
    
% features!!!
%F = [ I(1) Ih(30) G(1) Gh(20) Fr(1) Fh(30) ];
s = [ 361 30 1 20 1 30];

D = zeros(Dlength, sum(s), 'single');
L = zeros(Dlength, 1);
count = 0;

for i = 1:length(files)
    
    filename = [img_folder d_img(i).name];
    disp(['processing ' filename]);
    
    I = imread(filename);
    Ipad = padarray(I, PADSIZE, 'symmetric', 'both'); 
    
    % gradient
    [GN OR ] = gradientEstimate(Ipad, 3);
    
    % frangi 
    opts.FrangiScaleRatio = .5;
    opts.FrangiScaleRange = [.5 2];
    Fr = FrangiFilter2D(double(Ipad), opts);
    
    pts = datapoints{i};  %#ok<*USENS>
    l = labels{i}; 
    
    Idisp = gray2rgb(Ipad);
    Iedges = 0: (255-0)/(30-1): 255;
    GNedges = [0:(50-0)/(19-1):50 300];
    FRedges = [0:(.1-0)/(29-1):.1 1];
    
    % define the patch inds
   
    for j = 1:size(pts,1)
        pt = pts(j,:) + OFFSET;
        p = Ipad( pt(1)-WINDOW(1) : pt(1)+WINDOW(1), pt(2)-WINDOW(2):pt(2)+WINDOW(2));
        g = GN(pt(1)-WINDOW(1) : pt(1)+WINDOW(1), pt(2)-WINDOW(2):pt(2)+WINDOW(2));
        f = Fr(pt(1)-WINDOW(1) : pt(1)+WINDOW(1), pt(2)-WINDOW(2):pt(2)+WINDOW(2));

        if l(j) == 1
            Idisp(pt(1),pt(2),:) = [255 0 0];
        else
            Idisp(pt(1),pt(2),:) = [0 255 0];
        end
        
        %keyboard;
        
        count = count + 1;
        %D(count,1) = Ipad(pt(1), pt(2));
        D(count,1:361)                      = p(:);
        D(count,s(1)+1:s(1)+s(2))           = histc(p(:), Iedges);
        D(count,sum(s(1:2))+1:sum(s(1:3)))  = GN(pt(1), pt(2));
        D(count,sum(s(1:3))+1:sum(s(1:4)))  = histc(g(:), GNedges);
        D(count,sum(s(1:4))+1:sum(s(1:5)))  = Fr(pt(1), pt(2));
        D(count,sum(s(1:5))+1:sum(s(1:6)))  = histc(f(:), FRedges);
        
    	L(count) = l(j);
    end
        
    %figure; hold on;
    imshow(Idisp); pause(.1); drawnow;
    set(gca, 'Position', [0 0 1 1])
    

    %keyboard;
    
end

pathstr = [pwd '/'];
filename = [pathstr 'D2.mat'];
save(filename, 'D', 'L');
disp(['...saved ' filename]);


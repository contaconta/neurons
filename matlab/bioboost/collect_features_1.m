function collect_features_1()

ann_folder = 'phase/annotations/';
img_folder = 'phase/images/';
addpath(ann_folder);
addpath(img_folder);
%%

%%load DATA_LOCS.mat
files = [146];
datapoints{1} = [[1; 2] [3; 4]];
labels{1} = [1 2];

%%
% add path to frangi filter
addpath('/osshare/Work/neurons/matlab/toolboxes/frangi_filter_version2a/');

d_img = dir([img_folder '*.png']);

PADSIZE = [10 10];  OFFSET = PADSIZE; %#ok<*NASGU>
WINDOW = [9 9];

Dlength = 0;
for i = 1:length(files);
    Dlength = Dlength + size(datapoints{i},1);
end
    
% features!!!
%F = [ I(1) Ih(30) G(1) Gh(20) Fr(1) Fh(30) ];

D = zeros(Dlength, 1+30+1+20+1+30);
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
        
        count = count + 1;
        D(count,1) = Ipad(pt(1), pt(2));
        D(count,2:31) = histc(p(:), Iedges);
        D(count,32) = GN(pt(1), pt(2));
        D(count,33:52) = histc(g(:), GNedges);
        D(count,53) = Fr(pt(1), pt(2));
        D(count,54:83) = histc(f(:), FRedges);
        
    	L(count) = l(j);
    end
        
    %figure; hold on;
    imshow(Idisp); pause(1); drawnow;
    set(gca, 'Position', [0 0 1 1])
    

    %keyboard;
    
end

pathstr = [pwd '/'];
filename = [pathstr 'D1.mat'];
save(filename, 'D', 'L');
disp(['...saved ' filename]);


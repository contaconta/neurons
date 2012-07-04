function test_2(CLASSIFIER)

ann_folder = '/osshare/Work/Data/phase/annotations/';
img_folder = '/osshare/Work/Data/phase/images/';
load DATA_LOCS.mat

% add path to frangi filter
addpath('/osshare/Work/neurons/matlab/toolboxes/frangi_filter_version2a/');

%d_img = dir([img_folder '*.png']);

PADSIZE = [10 10];  OFFSET = PADSIZE; %#ok<*NASGU>
WINDOW = [9 9]; patchsize = 2*WINDOW + [1 1];

% Dlength = 0;
% for i = 1:length(files);
%     Dlength = Dlength + size(datapoints{i},1);
% end
    
% features!!!
%F = [ I(1) Ih(30) G(1) Gh(20) Fr(1) Fh(30) ];
s = [ 361 30 1 20 1 30];

D = zeros(1, sum(s), 'single');
%L = zeros(Dlength, 1);
count = 0;


% load the image    
filename = [img_folder 'neuron-0151.png'];
disp(['processing ' filename]);
I = imread(filename);
Ipad = padarray(I, PADSIZE, 'symmetric', 'both'); 

% load the annotation
afile = [ann_folder '151.png'];
A = imread(afile);
Apad = padarray(A, PADSIZE, 'symmetric', 'both');
L = zeros(size(Apad,1), size(Apad,2));
L(Apad(:,:,1) > 200) = 1;
L(L == 0) = -1;

%% precompute features=

% gradient
[GN OR ] = gradientEstimate(Ipad, 3);

% frangi 
opts.FrangiScaleRatio = .5;
opts.FrangiScaleRange = [.5 2];
Fr = FrangiFilter2D(double(Ipad), opts);

% histogram edges
Iedges = 0: (255-0)/(30-1): 255;
GNedges = [0:(50-0)/(19-1):50 300];
FRedges = [0:(.1-0)/(29-1):.1 1];

Idisp = gray2rgb(Ipad);
E = zeros(size(Ipad));

%% loop through image

for r = PADSIZE(1):size(Ipad,1)-PADSIZE(2)
    disp(['...processing row ' num2str(r) ' / ' num2str(size(Ipad,1)-PADSIZE(2))]);
    
    for c = PADSIZE(1):size(Ipad,2)-PADSIZE(2)
        pt = [r c];
        p = Ipad( pt(1)-WINDOW(1) : pt(1)+WINDOW(1), pt(2)-WINDOW(2):pt(2)+WINDOW(2));
        g = GN(pt(1)-WINDOW(1) : pt(1)+WINDOW(1), pt(2)-WINDOW(2):pt(2)+WINDOW(2));
        f = Fr(pt(1)-WINDOW(1) : pt(1)+WINDOW(1), pt(2)-WINDOW(2):pt(2)+WINDOW(2));

        % construct the feature vector
        D(1,1:361)                      = p(:);
        D(1,s(1)+1:s(1)+s(2))           = histc(p(:), Iedges);
        D(1,sum(s(1:2))+1:sum(s(1:3)))  = GN(pt(1), pt(2));
        D(1,sum(s(1:3))+1:sum(s(1:4)))  = histc(g(:), GNedges);
        D(1,sum(s(1:4))+1:sum(s(1:5)))  = Fr(pt(1), pt(2));
        D(1,sum(s(1:5))+1:sum(s(1:6)))  = histc(f(:), FRedges);

        E(r,c) = AdaBoostClassify_mex(CLASSIFIER.inds, CLASSIFIER.thresh, CLASSIFIER.pol, CLASSIFIER.alpha, D);
        
%         if E(r,c) == 1
%             Idisp(pt(1),pt(2),:) = [255 0 0];
%         else
%             Idisp(pt(1),pt(2),:) = [0 255 0];
%         end
    end
end

E = E(PADSIZE(1)+1:size(Ipad,1)-PADSIZE(1), PADSIZE(2)+1:size(Ipad,2)-PADSIZE(2));
save E.mat E;

%figure; hold on;
%imshow(Idisp); pause(.1); drawnow;
%set(gca, 'Position', [0 0 1 1])




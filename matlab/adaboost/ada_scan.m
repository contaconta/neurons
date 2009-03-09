function grouped_detections = ada_scan(CASCADE, LEARNERS, I, varargin)
%
%  ada_scan(CASCADE, LEARNERS, I, 3, [1 3.3], 1.1, 'SCAN_MAT.mat');
%
%  step = 3, scan_lims = [1 3.3], cascade threshold = 1.1
%
%

DSIZE = [24 24];  BIG_SAFE_NUM = 50000;

step = varargin{1};
scale_lims = varargin{2};
threshold = varargin{3};
tempfile = varargin{4};

if ~isa(I, 'double');
    cls = class(I); 
    I = mat2gray(I, [0 double(intmax(cls))]); 
end

% convert to grayscale if necessary
if size(I,3) > 1; I = rgb2gray(I); end 


%% step 1: extract scanning windows and their locations

scales = scale_selection(I, DSIZE, 'limits', scale_lims);
scanlist = zeros(BIG_SAFE_NUM,4);  WIN.Images = zeros([DSIZE BIG_SAFE_NUM]);
n = 0;

for s = 1:length(scales)
    W = round(DSIZE(2)*(1/scales(s)));
    H = round((DSIZE(1)/DSIZE(2)) * W);
    
    for r = 1:max(1,round(step/scales(s))): size(I,1) - H
        for c = 1:max(1,round(step/scales(s))): size(I,2) - W
            n = n + 1;
            scanlist(n,:) = [r c W H];
            Icrop = I(r:r+H-1, c:c + W-1);
            if ~isequal(size(Icrop), DSIZE)
                Icrop = imresize(Icrop, DSIZE);
            end
            WIN.Images(:,:,n) = Icrop;
        end
    end
end

WIN.Images = WIN.Images(:,:,1:n);
scanlist = scanlist(1:n,:);

%keyboard;


%% step 2: precompute features on the scanning windows
if exist(tempfile, 'file')
    load(tempfile);
else
    PREMAT = ada_cascade_precom(WIN, CASCADE, LEARNERS, tempfile);
    save(tempfile, 'PREMAT');
end
%load FIRSTSCAN.mat

%% step 3: classify the scanning windows

C = ada_test_classify_set(CASCADE, PREMAT, threshold);
detections = scanlist(C,:);


%% step 4: group detections
hits = zeros(size(detections));
% format for drawing lines and grouping
for k=1:size(detections,1);    
    r = detections(k,1);
    c = detections(k,2);
    r2 = r + detections(k,3);
    c2 = c + detections(k,4);
    hits(k,:) = [r c r2 c2];
end
grouped_detections = group_detections(hits);


%% step 5: plot image


figure; imshow(I); set(gca, 'Position', [0 0 1 1]);

%% display the grouped detections
for k=1:size(grouped_detections,1); 
    r = grouped_detections(k,1);
    c = grouped_detections(k,2);
    r2 = grouped_detections(k,3);
    c2 = grouped_detections(k,4);
    a(k) = line([c c2 c2 c2 c2 c c c], [r r r r2 r2 r2 r2 r]);
    set(a(k), 'Color', [0 1 0], 'LineWidth', 2, 'LineStyle', '-');
end

% %% display individual detections
% for k = 1:size(hits,1);
%     r = hits(k,1);
%     c = hits(k,2);
%     r2 = hits(k,3);
%     c2 = hits(k,4);
%     b(k) = line([c c2 c2 c2 c2 c c c], [r r r r2 r2 r2 r2 r]);
%     set(b(k), 'Color', [0 1 0], 'LineWidth', 0.5, 'LineStyle', '-.');
% end


%keyboard;

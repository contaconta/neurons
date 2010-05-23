function [LABELS, DATA] = collect_negative_examples(N, IMSIZE, folder, varargin)


d = dir([folder '*.jpg']);


DATA = zeros(N, (IMSIZE(1)+1)*(IMSIZE(2)+1), 'single');
%DATA = zeros(N, prod( (IMSIZE + [1 1]) ), 'single');
LABELS = -1* ones(N,1);

N_i = floor(N/length(d));

count = 1;

for i = 1:length(d)-1
    
    img_filename = [folder d(i).name];  
    [pathstr,name,ext,versn] = fileparts(img_filename);    
    I = imread(img_filename);
    disp(['sampling ' num2str(N_i) ' examples from ' name  ext]);
    
    for j = 1:N_i
        
        D = sample_rect(I, IMSIZE);
        
        D = integralImage(D, 'outer');
    
        D = single(D(:));
        
        DATA(count,:) = D;
        
        
        count = count + 1;
%         figure(1);
%         imshow(D);
%         drawnow;
%         pause(0.006);
    end
 
    
end

% the last one, fill up to N
N_i = N - count + 1;
img_filename = [folder d(length(d)).name];  
[pathstr,name,ext,versn] = fileparts(img_filename);    
I = imread(img_filename);
disp(['sampling ' num2str(N_i) ' examples from ' name  ext]);

for j = 1:N_i        
    D = sample_rect(I, IMSIZE);
    D = integralImage(D, 'outer');
    D = single(D(:));
    DATA(count,:) = D;
    count = count + 1;
end



function data = sample_rect(I, IMSIZE)

MIN_H_W = min(IMSIZE);


hmean = sqrt(numel(I))/20;
wmean = sqrt(numel(I))/20;
sigma = sqrt(numel(I))/10;

h = sigma*randn(1) + hmean;
w = sigma*randn(1) + wmean;

% sample the height and width
h = round(max(MIN_H_W, h));
w = round(max(MIN_H_W, w));

% h = floor(IMSIZE(2)/2);
% w = floor(IMSIZE(1)/2);
    
% sample the upper left corner 
r = ceil( (size(I,1)-h)*rand);
c = ceil( (size(I,2)-w)*rand);

%r = h+ceil((size(I,1)-2*h)*rand);
%c = w+ceil((size(I,2)-2*w)*rand);
    
% get a patch around the location
%cmax = min(size(I,2),cmin+IMSIZE(2)-1);
%rmax = min(size(I,1),rmin+IMSIZE(1)-1);

% cmin = max(1, c-w);
% rmin = max(1, r-h);
% cmax = min(size(I,2),cmin+IMSIZE(2)-1);
% rmax = min(size(I,1),rmin+IMSIZE(1)-1);

%data = imresize(I(rmin:rmax,cmin:cmax), IMSIZE);

%disp([num2str([r c h w])]);

data = imresize(I(r:r+h, c:c+w), IMSIZE);
function [LABELS, DATA] = collect_positive_examples(N, IMSIZE, folder, varargin)


d = dir([folder '*.png']); 

if N > length(d)
    warning(['Too many examples requested. ' num2str(N) ' requested, only ' num2str(length(d)) ' exist. Collecting all ' num2str(length(d)) '.']); %#ok<WNTAG>
    N = length(d);
end

inds = randsample(length(d), N)';


DATA = zeros(N, (IMSIZE(1)+1)*(IMSIZE(2)+1), 'single');
LABELS = ones(N,1);

count = 1;

for i = inds
    
    img_filename = [folder d(i).name];
    
    I = imread(img_filename);
    
    % NOTE:  WE MAY WANT TO DO INTENSITY NORMALIZATION HERE!
    
    % check that the sizes match
    if ~isequal(IMSIZE, size(I))
        I = imresize(I, IMSIZE);
    end
    
    I = integralImage(I, 'outer');
    
    I = single(I(:));
    
    DATA(count,:) = I;
    count = count + 1;
end

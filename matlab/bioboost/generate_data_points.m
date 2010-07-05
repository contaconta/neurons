

ann_folder = '/osshare/Work/Data/phase/annotations/';
img_folder = '/osshare/Work/Data/phase/images/';

N = 200000;     % negative examples
P = 20000;      % positive examples

d = dir([ann_folder '*.png']);
d_img = dir([img_folder '*.png']);

files = {};
datapoints = {};
labels = {};
nsum = 0; psum = 0;


for i = 1:length(d)-2  % do not use the last 2 images, save for testing!
    
    if i < length(d)-2
        n = N/(length(d)-2);  nsum = nsum + n;
        p = P/(length(d)-2);  psum = psum + p;
    else
        n = N - nsum;
        p = P - psum;
    end
    
    
    filename = [ann_folder d(i).name];
    files{i} = filename; %#ok<*SAGROW>
    
    A = imread(filename);
    
    POS = find(A(:,:,1) > 0);
    NEG = find(A(:,:,1) == 0);
    
    % randomly sample positive points
    inds_p = randsample(POS, p);
    [r_p c_p] = ind2sub([size(A,1),size(A,2)], inds_p);
    labels_p = ones(size(inds_p,1),1);
    
    % randomly sample negative points
    inds_n = randsample(NEG, n);
    [r_n c_n] = ind2sub([size(A,1),size(A,2)], inds_n);
    labels_n = -1 * ones(size(inds_n,1),1);
    
    datapoints{i} = [r_p c_p; r_n c_n];
    labels{i} = [labels_p; labels_n];
    
    
    % display;
    I = imread([img_folder d_img(i).name]);
    maskn = zeros(size(I));
    maskn(inds_n) = 1;
    I1 = imoverlay(I, maskn, 'color', [1 0 0], 'bright');
    maskp = zeros(size(I));
    maskp(inds_p) = 1;
    I1 = imoverlay(I1, maskp, 'color', [0 1 0], 'bright');
    imshow(I1);
    
    pause(1); drawnow;
end

save DATA_LOCS.mat labels datapoints files;


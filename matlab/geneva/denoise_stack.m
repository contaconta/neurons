function denoise_stack(source)

addpath('./bm3d/')

d = dir([source '*.tif']);

% check if image stacks exist
if isempty(d)
    error('no image stacks found');
else
    T = length(d);
end


MAX_SIZE = 25000000;



for t = 1:T
    
    fname = [source d(t).name];
    disp(['   reading: ' fname]);
    
    % read the tiff, clean it
    V = readMultiPageTiff(fname);
    
    disp(['   denoising: ' fname]);
    % denoise the image
    
    % flatten the volume
    [d1 d2 d3] = size(V);
    Vflat = reshape(V, d1, d2*d3);
    clear V;
    
    N = ceil(numel(Vflat)/MAX_SIZE);
    C = round((d2*d3)/N);
   
    t0 = tic;
    fprintf('  [completed out of %d]: ', N);
    for i = 1:N
        if i ~= N
            rlims = [1 d1];
            clims = [((C*(i-1)) +1) C*i];
        end
        if i == N
            rlims = [1 d1];
            clims = [((C*(i-1)) +1) size(Vflat,2)];
        end
        I = im2double(Vflat(rlims(1):rlims(2), clims(1):clims(2)));
        [NA, Idenoised] = BM3D(1, I, 25, 'lc');
        I2 = gray2ind(Idenoised, 255);
        fprintf('%d ', i);

        Vflat(rlims(1):rlims(2), clims(1):clims(2)) = I2;
    end
    pause(0.25);
    toc(t0);
    %fprintf('\n');
    
    V = reshape(Vflat, [d1 d2 d3]);
    clear Vlfat;
    
%     MIP = max(V, [], 3);
%     figure; imshow(MIP);
%     keyboard;
    
    % write the cleaned copy
    disp(['   overwriting: ' fname]);
    writeMultiPageTiff(V, fname);
    
end

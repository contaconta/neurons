function V2reg = register3DStacks(V1, V2)
%
%   V2reg = register3DStacks(V1, V2)
%   Registers V2 to V1.
%
%



%% step 1: compute the gaussian pyramids
REDUCTION_FACTOR    = 2;
SMOOTH_XY           = 1;
SMOOTH_Z            = 0.4;
DIFF_XY             = 0.4;   %1;  % 1 will produce [7 7 3]
DIFF_Z              = 0.4;
PYRAMID_LEVELS      = 4;
interpolation       = 'cubicXY';


% initialize the pyramids
P1_o = cell([PYRAMID_LEVELS 1]);    P2_o = P1_o;
P1_dx = cell([PYRAMID_LEVELS 1]);   P2_dx = P1_dx;
P1_dy = cell([PYRAMID_LEVELS 1]);   P2_dy = P1_dy;
P1_dz = cell([PYRAMID_LEVELS 1]);   P2_dz = P1_dz;
tic;

% generate 3D gaussian derivative kernels
s3 = gausskernel0_3D(SMOOTH_XY, SMOOTH_Z);
[dx dy dz] = gausskernel1_3D(DIFF_XY, DIFF_Z);
disp(['   smoothing kernel size = [' num2str(size(s3)) ']']);
disp(['   difference kernel size = [' num2str(size(dx)) '], ' interpolation ' interpolation']);

% create the base levels of the pyramids
fprintf('   computing Gaussian Pyramids:');
% fprintf(' level 1'); pause(0.01);
% V1 = imfilter(V1, s3, 'symmetric');         V2 = imfilter(V2, s3, 'symmetric');
% P1_o{1}  = V1;                              P2_o{1} = V2;
% P1_dx{1} = imfilter(V1, dx, 'symmetric');   P2_dx{1} = imfilter(V2, dx, 'symmetric');
% P1_dy{1} = imfilter(V1, dy, 'symmetric');   P2_dy{1} = imfilter(V2, dy, 'symmetric');
% P1_dz{1} = imfilter(V1, dz, 'symmetric');   P2_dz{1} = imfilter(V2, dz, 'symmetric');

% create upper levels of the pyramids
for i = 2:PYRAMID_LEVELS
    fprintf(' level %d', i); pause(0.01);
    V1 = reduceVolume(V1, REDUCTION_FACTOR, 'cubicXY');  % 'downsampleXY', 'linearXY'
    V2 = reduceVolume(V2, REDUCTION_FACTOR, 'cubicXY');
    
    V1 = imfilter(V1, s3, 'symmetric');         V2 = imfilter(V2, s3, 'symmetric');
    P1_o{1}  = V1;                              P2_o{1} = V2;
    P1_dx{i} = imfilter(V1, dx, 'symmetric');   P2_dx{i} = imfilter(V2, dx, 'symmetric');
    P1_dy{i} = imfilter(V1, dy, 'symmetric');   P2_dy{i} = imfilter(V2, dy, 'symmetric');
    P1_dz{i} = imfilter(V1, dz, 'symmetric');   P2_dz{i} = imfilter(V2, dz, 'symmetric');
end
disp(' '); toc;



%% step 2: magic!



%% ======== temporary, set output to input ========
V2reg = V2;




keyboard;



function V = reduceVolume(V, factor, method)



switch method 
    case 'downsampleXY'
        V = V(1:factor:end, 1:factor:end, :);
    case 'downsampleXYZ'
        V = V(1:factor:end, 1:factor:end, 1:factor:end);
    case 'cubicXY'
        I = imresize(V(:,:,1), 1/factor, 'nearest');
        V2 = zeros(size(I,1), size(I,2), size(V,3));
        for z = 1:size(V,3)
            V2(:,:,z) = imresize(V(:,:,z), 1/factor, 'bicubic'); 
        end
        V = V2;
    case 'linearXY'
        I = imresize(V(:,:,1), 1/factor, 'nearest');
        V2 = zeros(size(I,1), size(I,2), size(V,3));
        for z = 1:size(V,3)
            V2(:,:,z) = imresize(V(:,:,z), 1/factor, 'bilinear'); 
        end
        V = V2;
        
    %case 'nearest'
    %    resamp = makeresampler('nearest', 'symmetric');
    %case 'cubic'
    % 
    otherwise
        error('downsample method not specified');
end
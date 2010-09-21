function V = readMultiPageTiff(fname)


info = imfinfo(fname);

W = info(1).Width;
H = info(1).Height;
Z = numel(info);



if info(1).BitDepth == 16
    V = zeros(H,W,Z, 'uint16');
elseif info(1).BitDepth == 8
    V = zeros(H,W,Z, 'uint8');
else
    V = zeros(H,W,Z, 'single');
end


for k = 1:Z
    I = imread(fname, k, 'Info', info);
    V(:,:,k) = I;
end


% check if we need to correct for microscope acquisition weirdness
if max(V(:)) == 8288;
    disp('...correcting for microscope acquisition weirdness.');
    V = fix_alan_acquire(V);
end



function mv = loadMovie(Gfolder)


Gfiles = dir([Gfolder '*.TIF']);
IntensityAjustmentGreen.MED = 2537;
IntensityAjustmentGreen.STD = 28.9134;
IntensityAjustmentGreen.MAX = 11234;
if ~exist('TMAX', 'var'); TMAX =  length(Gfiles); end; % number of time steps
if TMAX~=length(Gfiles)
   disp(['problem with data in directory: ' Gfolder]);
   return;
end
[Green, Green_Original] = trkReadImagesAndNormalize(TMAX, Gfolder, IntensityAjustmentGreen);
mv = cell(size(Green));
B = zeros(size(Green{1},1), size(Green{1},2));
TMAX = length(Green);
parfor t = 1:TMAX
    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    mv{t} = I;
end
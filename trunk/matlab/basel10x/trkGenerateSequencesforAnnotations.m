function [] = trkGenerateSequencesforAnnotations(folder, resultsFolder, SeqIndexStr)

% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.TIF']);
Rfiles = dir([Rfolder '*.TIF']);

if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps

R = trkReadImages(TMAX, Rfolder);
G = trkReadImages(TMAX, Gfolder);
RG = cell(size(R));

if TMAX~=length(Gfiles)
   disp(['problem with data in directory: ' folder]);
   return;
end

parfor t = 1:TMAX
    R{t} = uint8(255*mat2gray(double(R{t})));
    G{t} = 255-uint8(255*mat2gray(double(G{t})));
    RG{t} = zeros(size(R{t}, 1), size(R{t}, 2), 3, 'uint8');
    RG{t}(:, :, 1) = R{t};
    RG{t}(:, :, 2) = G{t};
end

movfile = [SeqIndexStr 'Original'];
trkMovie(RG, resultsFolder, resultsFolder, movfile); fprintf('\n');
movfile = [SeqIndexStr 'OriginalRed'];
trkMovie(R, resultsFolder, resultsFolder, movfile); fprintf('\n');
movfile = [SeqIndexStr 'OriginalGreen'];
trkMovie(G, resultsFolder, resultsFolder, movfile); fprintf('\n');

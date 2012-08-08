function [] = trkGenerateSequencesforAnnotations(folder, resultsFolder, SeqIndexStr)

% define the folder locations and filenames of the images
Gfolder = [folder 'green/'];
Rfolder = [folder 'red/'];
Gfiles = dir([Gfolder '*.TIF']);%#ok
Rfiles = dir([Rfolder '*.TIF']);

if ~exist('TMAX', 'var'); TMAX =  length(Rfiles); end; % number of time steps

R = trkReadImages(TMAX, Rfolder);
G = trkReadImages(TMAX, Gfolder);
RG = cell(size(R));

for t = 1:TMAX
    Ilist = double(R{t}(:));
    
    [h,x] = hist(Ilist,1000);
    cumulative_hist = cumsum(h);
    cumulative_hist = cumulative_hist / max(cumulative_hist);
%     IntThreshold = min(x(cumulative_hist > 0.20));
%     Ilist = max(R{t}- IntThreshold, 0);
%     IntThreshold = min(x(cumulative_hist > 0.99));
%     Ilist = min(R{t}, IntThreshold);
%     R{t} = 255-uint8(255*mat2gray(double(Ilist)));
%     
%     
%     Ilist = double(G{t}(:));
%     
%     [h,x] = hist(Ilist,1000);
%     cumulative_hist = cumsum(h);
%     cumulative_hist = cumulative_hist / max(cumulative_hist);
%     IntThreshold = min(x(cumulative_hist > 0.20));
%     Ilist = max(G{t}- IntThreshold, 0);
%     IntThreshold = min(x(cumulative_hist > 0.995));
%     Ilist = min(Ilist, IntThreshold);
%     G{t} = uint8(255*mat2gray(double(Ilist)));
    
    R{t} = uint8(255*mat2gray(double(R{t})));
    G{t} = 255-uint8(255*mat2gray(double(G{t})));
    RG{t} = zeros(size(R{1}, 1), size(R{1}, 2), 3, 'uint8');
    RG{t}(:, :, 1) = R{t};
    RG{t}(:, :, 2) = G{t};
end

movfile = [SeqIndexStr 'Original'];
trkMovie(RG, resultsFolder, resultsFolder, movfile); fprintf('\n');

% R = trkReadImages(TMAX, Rfolder);
% G = trkReadImages(TMAX, Gfolder);
% for t = 1:TMAX
%     R{t} = uint8(255*mat2gray(double(R{t})));
%     G{t} = uint8(255*mat2gray(double(G{t})));
% end
movfile = [SeqIndexStr 'OriginalRed'];
trkMovie(R, resultsFolder, resultsFolder, movfile); fprintf('\n');
movfile = [SeqIndexStr 'OriginalGreen'];
trkMovie(G, resultsFolder, resultsFolder, movfile); fprintf('\n');

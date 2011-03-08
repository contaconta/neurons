function trkRemakeMovie(R, imgRootFolder, destFolder, keepFramesFlag)

% argument checks
if nargin < 4
    keepFramesFlag = 1;
end
if ~strcmp(imgRootFolder(end), '/')
    imgRootFolder = [imgRootFolder '/'];
end
if ~strcmp(destFolder(end), '/')
    destFolder = [destFolder '/'];
end

%% retrieve the images
imgFolder = [imgRootFolder R.GlobalMeasures.Date '/' R.GlobalMeasures.AssayPosition '/green/'];
Gfiles = dir([imgFolder '*.tif']);
mv = getImgFiles(imgFolder, Gfiles, R.GlobalMeasures.Length);

%% define a colormap for rendering
cols1 = jet(6);
cols1 = cols1(randperm(6),:);
cols2 = jet(8);
cols2 = cols2(randperm(8),:);
cols3 = jet(1000);
cols3 = cols3(randperm(1000),:);
colors = [cols1; cols2; cols3];

%% draw our results on top of the images
DISPLAY_FIGURES = 1;
%BLANK = zeros(size(mv{1}));
%mv = trkRenderImages3(1,R.GlobalMeasures.Length,R.GlobalMeasures.Date,R.GlobalMeasures.AssayPosition,R.GlobalMeasures.Label,colors,mv,R.Dlist,BLANK,R.FILAMENTS,R.Soma,R.tracks,R.D,DISPLAY_FIGURES);
mv = trkRenderImages4(1,R.GlobalMeasures.Length,R,colors,mv,DISPLAY_FIGURES);

%% write the movie
movfile = [  R.GlobalMeasures.Date '_' R.GlobalMeasures.AssayPosition '.avi'];
trkMovie(mv, destFolder, destFolder, movfile, ~keepFramesFlag); disp(' ');

disp('');
disp(['...encoded ' destFolder movfile]);

%keyboard;












function mv = getImgFiles(Gfolder, Gfiles, TMAX)


for t = 1:TMAX
    G = imread([Gfolder Gfiles(t).name]);
    
    if t == 1
        lims = stretchlim(G);
    end
    G8bits = trkTo8Bits(G, lims);
    
    % make an output image
    Ir = mat2gray(G8bits);
    I(:,:,1) = Ir;
    I(:,:,2) = Ir;
    I(:,:,3) = Ir;
    
    mv{t} = I;  %#ok<AGROW>
end


function J = trkTo8Bits(I, lims)


%lims = stretchlim(I);         
J = imadjust(I, lims, []); 
J = uint8(J/2^8);

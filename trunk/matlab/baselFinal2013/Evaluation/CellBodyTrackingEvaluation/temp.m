clear all;
seq = 3;

folder = sprintf('/home/ksmith/data/sinergia_evaluation/Selection10x/%03d/', seq);
seqFile = sprintf('/home/ksmith/data/sinergia_evaluation/Detections10x/%03d.mat', seq);
load(seqFile);

Gfolder = [folder 'green/'];
Gfiles = dir([Gfolder '*.TIF']);
IntensityAjustmentGreen.MED = 2537;
IntensityAjustmentGreen.STD = 28.9134;
IntensityAjustmentGreen.MAX = 11234;
if ~exist('TMAX', 'var'); TMAX =  length(Gfiles); end; % number of time steps
if TMAX~=length(Gfiles)
   disp(['problem with data in directory: ' folder]);
   return;
end
[Green, Green_Original] = trkReadImagesAndNormalize(TMAX, Gfolder, IntensityAjustmentGreen);


mv = cell(size(Green));

B = zeros(size(Green{1},1), size(Green{1},2));
TMAX = length(Green);
parfor t = 1:TMAX
    %I = mv{t};
    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;

    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    mv{t} = I;
end


cols = color_list();
B = zeros(size(Green{1},1), size(Green{1},2));
for i = 1:numel(Sequence.TrackedCells)
    fprintf('rendering cell %d/%d\n', i, numel(Sequence.TrackedCells));
    
    for t = 1:numel(Sequence.TrackedCells(i).TimeStep)
        
        framenumber = Sequence.TrackedCells(i).TimeStep(t).Time;
        I = mv{framenumber};
        Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);

        ID = i;  %Sequence.TrackedCells(i).TimeStep(t).ID;
        color = cols(ID,:);
    
        SomaM = B > Inf;
        SomaM(Sequence.TrackedCells(i).TimeStep(t).SomaPixelIdxList) = 1;
    
        SomaP = bwmorph(SomaM, 'remove');
        SomaP = bwmorph(SomaP, 'dilate');
        SomaP = bwmorph(SomaP, 'thin',1);
        Ir(SomaP) = max(0, color(1) - .2);
        Ig(SomaP) = max(0, color(2) - .2);
        Ib(SomaP) = max(0, color(3) - .2);
        
        % color the nucleus
        Ir(Sequence.TrackedCells(i).TimeStep(t).NucleusPixelIdxList) = color(1);
        Ig(Sequence.TrackedCells(i).TimeStep(t).NucleusPixelIdxList) = color(2);
        Ib(Sequence.TrackedCells(i).TimeStep(t).NucleusPixelIdxList) = color(3);
            
        I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;

        mv{framenumber} = I;
    end
end

for t = 1:numel(mv)
    I = mv{t};
    I = uint8(255*I);
    blk = [80 80 80];
    num_txt = sprintf('tracking_evaluation_sequence_%03d', seq);
    
    I=trkRenderText(I,num_txt, blk, [10, 50], 'bnd2', 'left');
    mv{t} = I;
end

for i = 1:numel(Sequence.TrackedCells)
    fprintf('rendering text %d/%d\n', i, numel(Sequence.TrackedCells));
    
    for t = 1:numel(Sequence.TrackedCells(i).TimeStep)
        framenumber = Sequence.TrackedCells(i).TimeStep(t).Time;
        I = mv{framenumber};

        ID = i;  %Sequence.TrackedCells(i).TimeStep(t).ID;
        color = cols(ID,:);
        
        coloroffset = [-.1 -.1 -.1];   

        col = max(color+coloroffset, 0);
        rloc = max(1,Sequence.TrackedCells(i).TimeStep(t).SomaCentroid(2) -20); %- 30);
        cloc = max(1,Sequence.TrackedCells(i).TimeStep(t).SomaCentroid(1) +15); %+ 20);
        
%         str = sprintf('ID=%d', ID);   
        str = sprintf('%d', ID);
        I=trkRenderText(I,str, floor(255*col), [rloc, cloc], 'bnd2', 'left');    

        mv{framenumber} = I;
    end
end


outputFolder = '/home/ksmith/data/sinergia_evaluation/output/';
movfile = sprintf('%03d', seq);
trkMovie(mv, outputFolder, outputFolder, movfile); fprintf('\n');
fprintf('wrote to %s%s.mp4\n', outputFolder,movfile);

% keyboard;




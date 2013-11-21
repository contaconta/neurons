clear all; close all; %clc;
%%
prerequisites;
if 0 
    matlabpool local 
end
overlappingTolerance = 0.5;
isDetectionDone = true;
%%
Magnification = '10x';
dataRootDirectory    = ['/Users/feth/GoogleDrive/Sinergia/GT' Magnification '/Dynamic/'];
ConvertedGTRootDir   = ['/Users/feth/GoogleDrive/Sinergia/GT' Magnification '/Dynamic_matlab/'];
RawRootDataDirectory = ['/Users/feth/GoogleDrive/Sinergia/Selection' Magnification '/'];
DetectionDirectory   = ['/Users/feth/GoogleDrive/Sinergia/Detections' Magnification '/'];
if(~exist(DetectionDirectory, 'dir'))
    mkdir(DetectionDirectory);
end
%%
listOfGTSeq = dir(dataRootDirectory);
AllTruePositives = [];
disp('========================================')
seqIdx = 1;
for i = 1:length(listOfGTSeq)
   if(listOfGTSeq(i).isdir && ~isempty(str2num(listOfGTSeq(i).name)) ) %#ok
       inputSeqDirToProcess = [RawRootDataDirectory listOfGTSeq(i).name '/'];
%        disp('========================================')
%        disp(['Original Sequence: ' inputSeqDirToProcess]);       
       detectionsFileName = [DetectionDirectory listOfGTSeq(i).name 'CellBodyDetections'];
       if ~isDetectionDone
           [Nuclei, Somata, Cells, CellsList] =  trkDetectNucleiSomataForEvaluation(inputSeqDirToProcess, Magnification);
           save(detectionsFileName, 'Cells', 'CellsList');
       else
           load(detectionsFileName);
       end
       
       GTFileName = [ConvertedGTRootDir listOfGTSeq(i).name];
       
%        disp(['Annotation file: ' GTFileName]);
       load(GTFileName);
       
       [TruePositives, FalsePositives] = trkEvaluateDetectionsSomata(Cells, CellsList, AnnotatedTrackedCells, overlappingTolerance);
       FalsePositivesPerImage{seqIdx}  = FalsePositives;%#ok
       TruePositivesPerImage{seqIdx}   = TruePositives;%#ok
       seqIdx = seqIdx + 1;
       AllTruePositives    = [AllTruePositives TruePositives];%#ok
%        disp('========================================')
%        keyboard;
   end
end
%%
disp(['avg score :' num2str(100*numel(find(AllTruePositives > 0)) / numel(AllTruePositives)) '%']);
for i =1:length(TruePositivesPerImage)
    disp(['score for image ' int2str(i) ' is :' num2str(100*numel(find(TruePositivesPerImage{i} > 0)) / numel(TruePositivesPerImage{i})) '%']);
end
%%
seqIdx                        = 3;
sequence_index_for_inspection = 13;

inputSeqDirToProcess          = [RawRootDataDirectory sprintf('%03d', sequence_index_for_inspection)];
detectionsFileName            = [DetectionDirectory   sprintf('%03d', sequence_index_for_inspection)];
GTFileName                    = [ConvertedGTRootDir   sprintf('%03d', sequence_index_for_inspection)];

load(detectionsFileName);
load(GTFileName);

for image_index = 1:97
    Im = load_image_from_sequence(inputSeqDirToProcess, 'red', image_index);
    Im = max(Im(:)) - Im;
    [Im, GT, Detections, TruePositives, FalsePositives, FalseNegatives] = trkRenderImageAndDetections(Im, Cells, CellsList, AnnotatedTrackedCells, TruePositivesPerImage{seqIdx}, FalsePositivesPerImage{seqIdx},  image_index);
    [~, num] = bwlabel(FalseNegatives);
    nb_FN_perFrame(image_index) = num;%#ok
end
%% code for macking figures
image_index = 1;
Im = load_image_from_sequence(inputSeqDirToProcess, 'green', image_index);
Im = max(Im(:)) - Im;

[Im, GT, Detections, TruePositives, FalsePositives, FalseNegatives] = trkRenderImageAndSomataDetections(Im, Cells, CellsList, AnnotatedTrackedCells, TruePositivesPerImage{seqIdx}, FalsePositivesPerImage{seqIdx},  image_index);

figure(1); clf; 
imshow(Im, [], 'InitialMagnification', 200);
pause;
print2im('OriginalGreen.png');


figure(1); clf; 

imshow(Im, [], 'InitialMagnification', 200);
blue = cat(3, zeros(size(Im)), zeros(size(Im)), ones(size(Im)));
hold on;
h = imshow(blue);
hold off;
set(h, 'AlphaData', 0.5*GT);
pause;
print2im('SomataGTGreen.png');

figure(1); clf; 

imshow(Im, [], 'InitialMagnification', 200);
magenta = cat(3, ones(size(Im)), zeros(size(Im)), ones(size(Im)));
hold on;
h = imshow(magenta);
hold off;
set(h, 'AlphaData', 0.5*Detections);
    
pause;
print2im('SomataDetectionGreen.png');




%%
for image_index                   = 1:53;
    Im = load_image_from_sequence(inputSeqDirToProcess, 'red', image_index);
    Im = max(Im(:)) - Im;

    [Im, GT, Detections, TruePositives, FalsePositives, FalseNegatives] = trkRenderImageAndDetections(Im, Cells, CellsList, AnnotatedTrackedCells, TruePositivesPerImage{seqIdx}, FalsePositivesPerImage{seqIdx},  image_index);

    figure(1); clf; 
    ax(1) = subplot(3 , 2, 1);
    imshow(Im, [], 'InitialMagnification', 200);
    title(['Original Image : ' int2str(image_index)]);

    ax(2) = subplot(3, 2, 2);
    imshow(Im, [], 'InitialMagnification', 200);
    blue = cat(3, zeros(size(Im)), zeros(size(Im)), ones(size(Im)));
    hold on;
    h = imshow(blue);
    hold off;
    set(h, 'AlphaData', 0.5*GT);
    title('Ground truth annotations');


    ax(3) = subplot(3, 2, 3);
    imshow(Im, [], 'InitialMagnification', 200);
    magenta = cat(3, ones(size(Im)), zeros(size(Im)), ones(size(Im)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', 0.5*Detections);
    title('Autmatic detections');

    ax(4) = subplot(3, 2, 4);
    imshow(Im, [], 'InitialMagnification', 200);
    green = cat(3, zeros(size(Im)), ones(size(Im)), zeros(size(Im)));
    hold on;
    h = imshow(green);
    hold off;
    set(h, 'AlphaData', 0.5*TruePositives);
    title('True Positives');

    ax(5) = subplot(3, 2, 5);
    imshow(Im, [], 'InitialMagnification', 200);
    red = cat(3, ones(size(Im)), zeros(size(Im)), zeros(size(Im)));
    hold on;
    h = imshow(red);
    hold off;
    set(h, 'AlphaData', 0.9*FalsePositives);
    title('False Positive');


    ax(6) = subplot(3, 2, 6);
    imshow(Im, [], 'InitialMagnification', 200);
    red = cat(3, ones(size(Im)), zeros(size(Im)), zeros(size(Im)));
    hold on;
    h = imshow(red);
    hold off;
    set(h, 'AlphaData', 0.9*FalseNegatives);
    title('False Negatives');

    linkaxes(ax)
    pause;
end
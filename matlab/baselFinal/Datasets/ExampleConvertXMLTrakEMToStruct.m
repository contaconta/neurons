clear all; close all; clc;
%%
NumOfTrackedCells = 0;
NumOfAnnotatedNuclei = 0;
NumOfAnnotatedSomata = 0;

for idx = 1:1
    strIDx = sprintf('%03d', idx);
    disp('========================================')
    disp(strIDx)
    disp('========================================')
    DataRootDirectory = '/Users/feth/Google Drive/Sinergia/GT20x/Dynamic/';
    templateHeaderFile  = 'TemplateHeaderSimplified.xml';
    %%
    TrackedCells = getStructureFromTrakEM2XML(DataRootDirectory, idx, templateHeaderFile);
    NumOfTrackedCells = NumOfTrackedCells + numel(TrackedCells);
    for k = 1:numel(TrackedCells)
        NumOfAnnotatedNuclei = NumOfAnnotatedNuclei + numel(TrackedCells{k}.nucleus.listOfObjects.t2_area);
        if(isfield(TrackedCells{k}, 'soma'))
            NumOfAnnotatedSomata = NumOfAnnotatedSomata + numel(TrackedCells{k}.soma.listOfObjects.t2_area);
        end
    end
end
%%
disp(['Nb tracked cells is ' num2str(NumOfTrackedCells)]);
disp(['Nb annotated Nuclei is ' num2str(NumOfAnnotatedNuclei)]);
disp(['Nb annotated Somata is ' num2str(NumOfAnnotatedSomata)]);

%%
inputImage = [DataRootDirectory strIDx '/green.tif'];

I = readMultiPageTiff(inputImage);
%%
CellIdx = 1;
for detectiionIdx=1:10

    Time =  TrackedCells{CellIdx}.soma.listOfObjects.t2_area{detectiionIdx}.Time;
%     disp(num2str(Time));
    currentFrame = I(:,:, Time);
%     currentFrame(TrackedCells{CellIdx}.soma.listOfObjects.t2_area{detectiionIdx}.PixelIdxList) = max(currentFrame(:));
    figure(1); clf; imshow(currentFrame, []); hold on;
    plot(TrackedCells{CellIdx}.soma.listOfObjects.t2_area{detectiionIdx}.XX, TrackedCells{CellIdx}.soma.listOfObjects.t2_area{detectiionIdx}.YY, '-r')
    pause;
end
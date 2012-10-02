clear all; close all; clc;
%%
idx = 3;
strIDx = sprintf('%03d', idx);
DataRootDirectory = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/20xTrakEM2/';
templateHeaderFile  = 'TemplateHeaderSimplified.xml';
%%
TrackedCells = getStructureFromTrakEM2XML(DataRootDirectory, idx, templateHeaderFile);

inputImage = [DataRootDirectory strIDx '/red.tif'];


%%
I = readMultiPageTiff(inputImage);
%%
CellIdx = 3;
for detectiionIdx=1:4

    Time =  TrackedCells{CellIdx}.nucleus.listOfObjects.t2_area{detectiionIdx}.Time;
    disp(num2str(Time));
    figure(1); clf; imshow(I(:,:, Time), []); hold on;
    plot(TrackedCells{CellIdx}.nucleus.listOfObjects.t2_area{detectiionIdx}.XX, TrackedCells{CellIdx}.nucleus.listOfObjects.t2_area{detectiionIdx}.YY, '-r')
    pause;
end
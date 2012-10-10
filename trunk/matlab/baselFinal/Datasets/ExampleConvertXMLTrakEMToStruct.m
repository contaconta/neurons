clear all; close all; clc;
%%
idx = 2 ;
strIDx = sprintf('%03d', idx);
DataRootDirectory = '/Users/feth/tmp/';
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
%     disp(num2str(Time));
    figure(1); clf; imshow(I(:,:, Time), []); hold on;
    plot(TrackedCells{CellIdx}.nucleus.listOfObjects.t2_area{detectiionIdx}.XX, TrackedCells{CellIdx}.nucleus.listOfObjects.t2_area{detectiionIdx}.YY, '-r')
    pause;
end
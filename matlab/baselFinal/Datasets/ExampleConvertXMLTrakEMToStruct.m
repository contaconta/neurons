clear all; close all; clc;
%%
idx = 5;
strIDx = sprintf('%03d', idx);
DataRootDirectory = '/Users/feth/tmp/';
templateHeaderFile  = 'TemplateHeaderSimplified.xml';
%%
TrackedCells = getStructureFromTrakEM2XML(DataRootDirectory, idx, templateHeaderFile);
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
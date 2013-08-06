clear all; close all; clc;
%%
prerequisites;


%%
Magnification = '10x';
inputRawDataFile = '/home/fbenmans/SelectionStatic10X/10XStaticSelections.txt';
OutputFolder     = '/home/fbenmans/ProcessingsForEvaluation/NeuriteEvaluation/';

%%

fid = fopen(inputRawDataFile, 'r');

tline = fgets(fid);
i = 1;
while ischar(tline)
    disp(tline)
    C = textscan(tline, '%s \t %d %d');
    directoryName = [C{1}{1} num2str(C{2}) '/'];
    trkTracking(directoryName, OutputFolder, num2str(i), 'Eval', Magnification);
    tline = fgets(fid);
    i = i + 1;
end

fclose(fid);

clear all; close all; clc;
%% BrainBow

listOfInputDirs = {'BrainBow/gt/cropped2/',   'BrainBow/gt/cropped3/',...
                   'BrainBow/ip/cropped2/',   'BrainBow/ip/cropped3/',... 
                   'BrainBow/qmip/cropped2/', 'BrainBow/qmip/cropped3/'};

listOfDS_SWC    = {'BrainBow/gt/cropped2_ds/',   'BrainBow/gt/cropped3_ds/',...
                   'BrainBow/ip/cropped2_ds/',   'BrainBow/ip/cropped3_ds/',... 
                   'BrainBow/qmip/cropped2_ds/', 'BrainBow/qmip/cropped3_ds/'};
               
for i =1:length(listOfInputDirs)
    DownSampleSWCFilesDir(listOfInputDirs{i}, listOfDS_SWC{i});
end
%% BrightField

listOfInputDirs = {'N2/GT/',   'N7_cropped/gt/', ...
                   'N2/IP/',   'N7_cropped/ip/',... 
                   'N2/QMIP/', 'N7_cropped/qmip/'};

listOfDS_SWC    = {'N2/GT_ds/',   'N7_cropped/gt_ds/', ...
                   'N2/IP_ds/',   'N7_cropped/ip_ds/',... 
                   'N2/QMIP_ds/', 'N7_cropped/qmip_ds/'};
               
for i =1:length(listOfInputDirs)
    DownSampleSWCFilesDir(listOfInputDirs{i}, listOfDS_SWC{i});
end

%%
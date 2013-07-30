clear all; close all; clc;
%% trees toolbox required
run('../EvalForBrainBow/trees_toolbox/start_trees.m');

%%
listOfInputDirs = {'BrainBow/gt/cropped2_ds/',   'BrainBow/gt/cropped3_ds/',...
                   'BrainBow/ip/cropped2_ds/',   'BrainBow/ip/cropped3_ds/',... 
                   'BrainBow/qmip/cropped2_ds/', 'BrainBow/qmip/cropped3_ds/'};

listOfOutputObjs = {'BrainBow/gt_cropped2_ds.obj',   'BrainBow/gt_cropped3_ds.obj',...
                   'BrainBow/ip_cropped2_ds.obj',   'BrainBow/ip_cropped3_ds.obj',... 
                   'BrainBow/qmip_cropped2_ds.obj', 'BrainBow/qmip_cropped3_ds.obj'};

for i =1:length(listOfInputDirs)
    swcFilesDir   = listOfInputDirs{i};
    outputObjFile = listOfOutputObjs{i};
    [Graph, Branches] = convertSWCDir2Obj(swcFilesDir, outputObjFile);
end


%%
listOfInputDirs = {'N2/GT_ds/',   'N7_cropped/gt_ds/', ...
                   'N2/IP_ds/',   'N7_cropped/ip_ds/',... 
                   'N2/QMIP_ds/', 'N7_cropped/qmip_ds/'};

listOfOutputObjs = {'N2/GT_ds.obj',   'N7_cropped/gt_ds.obj', ...
                   'N2/IP_ds.obj',   'N7_cropped/ip_ds.obj',... 
                   'N2/QMIP_ds.obj', 'N7_cropped/qmip_ds.obj'};

for i =1:length(listOfInputDirs)
    swcFilesDir   = listOfInputDirs{i};
    outputObjFile = listOfOutputObjs{i};
    [Graph, Branches] = convertSWCDir2Obj(swcFilesDir, outputObjFile);
end

%%




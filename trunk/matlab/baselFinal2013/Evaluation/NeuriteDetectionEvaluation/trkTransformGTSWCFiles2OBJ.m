clear all; close all; clc;
%%

%%
isGTRescalingDone   = false;
isGTDownSamplingDone= false;
isGTConvertionDone  = false;

%% trees toolbox needed.
run('ThirdPartyCode/TREES1.15/start_trees.m')


%% Define the magnification and the data directories
Magnification = '10x';
RawRootDataDirectory = ['/Users/feth/Google Drive/Sinergia/StaticSelection' Magnification '/'];
GTDataDirectory      = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/StaticNeurites/'];
ConvertedGTRootDir   = ['/Users/feth/Google Drive/Sinergia/GT' Magnification '/StaticNeurites_obj/'];


%%
listOfGTimg = dir(GTDataDirectory);
disp('========================================')
imgIdx = 1;
for i = 1:length(listOfGTimg)
    if(listOfGTimg(i).isdir && ~isempty(str2num(listOfGTimg(i).name)) ) %#ok
        disp('========================================')
        disp(['========= processing GT ' listOfGTimg(i).name ' ========='])
        disp('========================================')
        % first, get the directory of the raw image and of the ground truth 
        inputImgDirToProcess = [RawRootDataDirectory listOfGTimg(i).name '/'];
        GTDir                = [GTDataDirectory      listOfGTimg(i).name '/'];
        GTRecaledDir         = [GTDataDirectory      listOfGTimg(i).name '_rescaled/'];
        GTDownSampledSWCDir  = [GTDataDirectory      listOfGTimg(i).name '_DS/'];
        
        % as the spacing of the images on which the ground truth tracing
        % have been modified to [h, h], with h = 0.35277777777777775, a
        % rescaling of the ground truth swc files is needed
        if(~isGTRescalingDone)
            h = 0.35277777777777775;
            % this value is obtained from the header of image files located
            % at /Google Drive/Sinergia/GT10x/StaticNeurites/%03d/%03d.nrrd
            rescaleSWCFilesDir(h, GTDir, GTRecaledDir);
        end
        % downsample the swc files if needed
        if(~isGTDownSamplingDone)
            DownSampleSWCFilesDir(GTRecaledDir, GTDownSampledSWCDir)
        end
        
        % convert ground truth to obj format for NetMets evaluation
        gtFileName           = [ConvertedGTRootDir listOfGTimg(i).name '.obj'];
        
        if(~isGTConvertionDone)
            convertSWCDir2Obj(GTDownSampledSWCDir, gtFileName);
        end
    end
end
%%

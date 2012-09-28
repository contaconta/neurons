clear all; close all; clc;
addpath('../');
%%
RootDirectorySelectionMovies = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/Selection20x/';
listOfMovies = {'003', '004'};
channels = {'red', 'green'};
OutputRootDirectory = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/20xTrakEM2/';
numberOfFrames = 97;
templateFile = 'Template.xml';

if(~exist(OutputRootDirectory, 'dir'))
    mkdir(OutputRootDirectory);
end

for i = 1:length(listOfMovies)
    
    
    outputdir = [OutputRootDirectory listOfMovies{i} '/'];
    if exist(outputdir, 'dir')
        rmdir(outputdir, 's');
    end
    
    if ~exist(outputdir, 'dir')
      mkdir(outputdir);  
    end
    
    
    for j = 1:length(channels)
        dirName = [RootDirectorySelectionMovies listOfMovies{i} '/' channels{j} '/'];
        
        listImages = dir(fullfile(dirName, '*.TIF'));
        names = {listImages.name};
        listImages = sort_nat(names);

        I = imread([dirName '/' listImages{1}]);
        Im = zeros([size(I, 1), size(I, 2), length(listImages)], 'uint16');
        
        if numberOfFrames~=length(listImages)
            error(['number of frames should be equal to ' num2str(numberOfFrames)]);
        end
        
        for k = 1:length(listImages)
            Im(:, :, k) = imread([dirName '/' listImages{i}]);
        end
        
        writeMultiPageTiff(Im, [outputdir channels{j} '.tif']);
        if strcmp(channels{j}, 'red')
           minRed = min(Im(:));
           maxRed = max(Im(:));
        elseif strcmp(channels{j}, 'green')
           minGreen = min(Im(:));
           maxGreen = max(Im(:));            
        end
    end
    A = xml2struct(templateFile);
    %% Do stuff
    xmlFileName = [RootDirectorySelectionMovies listOfMovies{i} '/' listOfMovies{i} '.xml'];
    A2 = A(2);
    B.trakEMCells = A;
    struct2xml(A1, xmlFileName);
end
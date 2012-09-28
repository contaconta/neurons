clear all; close all; clc;
addpath('../');
%%
RootDirectorySelectionMovies = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/Selection20x/';
AA = dir(RootDirectorySelectionMovies);
inc = 1;
for i =1:length(AA)
    if(length(AA(i).name) >2 && AA(i).isdir)
        listOfMovies{inc} = AA(i).name;%#ok
        inc = inc+1;
    end
end
channels = {'red', 'green'};
OutputRootDirectory = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/20xTrakEM2/';
numberOfFrames = 97;
templateFile        = 'Template.xml';
templateHeaderFile  = 'TemplateHeader.xml';

if(~exist(OutputRootDirectory, 'dir'))
    mkdir(OutputRootDirectory);
end

for i = 1:length(listOfMovies)
    disp(i)
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
            warning(['number of frames should be equal to ' num2str(numberOfFrames)]);
        end
        
        for k = 1:length(listImages)
            Im(:, :, k) = imread([dirName '/' listImages{k}]);
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
    A.trakem2.project.Attributes.title = [listOfMovies{i} '.xml'];
    for k =1:length(A.trakem2.t2_layer_set.t2_layer)
        A.trakem2.t2_layer_set.t2_layer{k}.t2_patch{1}.Attributes.max = num2str(maxRed);
        A.trakem2.t2_layer_set.t2_layer{k}.t2_patch{1}.Attributes.min = num2str(minRed);
        A.trakem2.t2_layer_set.t2_layer{k}.t2_patch{2}.Attributes.max = num2str(maxGreen);
        A.trakem2.t2_layer_set.t2_layer{k}.t2_patch{2}.Attributes.min = num2str(minGreen);
    end
    %% write to output xmlfile
    TxmlFileName = [outputdir listOfMovies{i} 'tmp.xml'];
    xmlFileName  = [outputdir listOfMovies{i} '.xml'];
    struct2xml(A, TxmlFileName);
    %% add header
    cmd_header = ['cat <(head -n1 ' TxmlFileName ') '  templateHeaderFile ' <(tail +2 ' TxmlFileName ') > ' xmlFileName];
    system(cmd_header);
    cmd_clean = ['rm ' TxmlFileName];
    system(cmd_clean);
%     xmlFileNameNoHeader  = [outputdir listOfMovies{i} 'NoHeader.xml'];
%     cmd_cleanXMLFromHeader = ['grep -F -x -v -f ' templateHeaderFile ' ' xmlFileName ' >> ' xmlFileNameNoHeader];
%     system(cmd_cleanXMLFromHeader);
    
end
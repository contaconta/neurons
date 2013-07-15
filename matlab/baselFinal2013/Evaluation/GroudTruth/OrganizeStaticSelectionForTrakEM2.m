clear all; close all; clc;
addpath('../');
%%
RootDirectorySelectionImages    = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/SelectionStatic20x/';
RootDirectoryOutput             = '/Users/feth/Documents/Work/Data/Sinergia/Olivier/20xStaticTrakEM2/';
if(~exist(RootDirectoryOutput, 'dir'))
   mkdir(RootDirectoryOutput);
end
AA = dir(RootDirectorySelectionImages);
inc = 1;
for i =1:length(AA)
    if(length(AA(i).name) >2 && AA(i).isdir)
        listOfImages{inc} = AA(i).name;%#ok
        inc = inc+1;
    end
end
channels = {'red', 'green'};
numberOfFrames = 97;
templateFile        = 'TemplateStatic.xml';
templateHeaderFile  = 'TemplateStaticHeader.xml';

for i = 1:length(listOfImages)
    disp(i)
    inputdir  = [RootDirectorySelectionImages listOfImages{i} '/'];
    outputdir = [RootDirectoryOutput listOfImages{i} '/'];
    if exist(outputdir, 'dir')
        rmdir(outputdir, 's');
    end
    if ~exist(outputdir, 'dir')
      mkdir(outputdir);  
    end
    
    
    
    for j = 1:length(channels)
        InputImageFileName      = [inputdir  channels{j} '.tif'];
        OutputImageFileName     = [outputdir channels{j} '.tif'];
        Im               = imread(InputImageFileName);
        Im = max(Im(:)) - Im;
        if strcmp(channels{j}, 'red')
           minRed = min(Im(:));
           maxRed = max(Im(:));
        elseif strcmp(channels{j}, 'green')
           minGreen = min(Im(:));
           maxGreen = max(Im(:));            
        end
        imwrite(Im, OutputImageFileName);
    end
    A = xml2struct(templateFile);
    %% Do stuff
    A.trakem2.project.Attributes.title = [listOfImages{i} '.xml'];

    A.trakem2.t2_layer_set.t2_layer.t2_patch{1}.Attributes.max = num2str(maxRed);
    A.trakem2.t2_layer_set.t2_layer.t2_patch{1}.Attributes.min = num2str(minRed);
    A.trakem2.t2_layer_set.t2_layer.t2_patch{2}.Attributes.max = num2str(maxGreen);
    A.trakem2.t2_layer_set.t2_layer.t2_patch{2}.Attributes.min = num2str(minGreen);

    %% write to output xmlfile
    TxmlFileName = [outputdir listOfImages{i} 'tmp.xml'];
    xmlFileName  = [outputdir listOfImages{i} '.xml'];
    struct2xml(A, TxmlFileName);
    %% add header
    cmd_header = ['cat <(head -n1 ' TxmlFileName ') '  templateHeaderFile ' <(tail +2 ' TxmlFileName ') > ' xmlFileName];
    system(cmd_header);
    cmd_clean = ['rm ' TxmlFileName];
    system(cmd_clean);
end
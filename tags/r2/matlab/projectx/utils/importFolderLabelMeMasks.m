addpath([pwd, '/../../toolboxes/LabelMeToolbox/'], '-begin');   % append the path to kevin's toolbox
disp('--------------------------------------------------------------------------');
disp('  THIS TOOL IS FOR IMPORTING IMAGES AND BINARY ANNOTATION MASK FILES ');
disp('  IMAGE AND ANNOTATION FILES MUST HAVE CORRESPONDING ORDER WITHIN EACH FOLDER.');
disp(' ');
SOURCEFOLDER = input('To import images, first create a folder containing ONLY images with the \nsame annotation. Provide the path to the folder (e.g. /home/source/)\nPATH: ', 's');
ANNFOLDER = input('\nThe annotations must appear in the same order as the images. Provide the\npath to the folder containing the annotation files (e.g. /home/annotation/)\nPATH: ', 's');
HOMEIMAGES = input('\nProvide the path to your LabelMe HOMEIMAGESFOLDER \n(e.g. /home/user/LabelMe/Images/)\nPATH: ', 's');
HOMEANNOTATIONS = input('\nProvide the path to your LabelMe HOMEANNOTATIONS \n(e.g. /home/user/LabelMe/Annotations/)\nPATH: ', 's');
LABELMEFOLDER = input(['\nProvide a new or existing folder name in what the images will be\nstored: ' HOMEIMAGES 'FOLDER_NAME\nFOLDER NAME: '], 's');
scene_description = input('\nPlease enter keywords for the scene description \n(e.g. street urban city)\nKEYWORDS: ', 's');
ann = input('\nType the annotation associated with these images.\nANNOTATION: ', 's');
disp('--------------------------------------------------------------------------');

if ~strcmp(SOURCEFOLDER(length(SOURCEFOLDER)), '/')
    SOURCEFOLDER(length(SOURCEFOLDER)+1) = '/';
end
if ~strcmp(ANNFOLDER(length(ANNFOLDER)), '/')
    ANNFOLDER(length(ANNFOLDER)+1) = '/';
end

if ~strcmp(HOMEIMAGES(length(HOMEIMAGES)), '/')
    HOMEIMAGES(length(HOMEIMAGES)+1) = '/';
end
if ~strcmp(HOMEANNOTATIONS(length(HOMEANNOTATIONS)), '/')
    HOMEANNOTATIONS(length(HOMEANNOTATIONS)+1) = '/';
end


% create the destination folder if it does not already exist
mkdir([HOMEIMAGES LABELMEFOLDER]);
mkdir([HOMEANNOTATIONS LABELMEFOLDER]);

% NOTE: IF SOURCE AND ANNOTATAION FOLDERS DO NOT APPEAR SAME IN DIR ORDER,
% THE CORRESPONDANCE BETWEEN IMAGES AND ANNOTATIONS CAN BE WRONG
% create a list of the images in the source folder
d1 = dir([SOURCEFOLDER '*.png']);
d2 = dir([SOURCEFOLDER '*.jpg']);
d = [d1; d2];
dann1 = dir([SOURCEFOLDER '*.png']);
dann2 = dir([SOURCEFOLDER '*.jpg']);
dann = [dann1; dann2];

clear prepend_string;
for i = 1:length(d)
    
    % copy the image to the destination folder
    FileName = d(i).name;
    srcFileName = [SOURCEFOLDER d(i).name];
    dstFileName = [HOMEIMAGES LABELMEFOLDER '/' FileName];
    
    AnnName = dann(i).name;
    
    % check to see if the destination file already exists, if so we have a
    % conflict we need to resolve by prepending a string to the filename
    if (exist(dstFileName, 'file'))
        if (~exist('prepend_string','var'))
            disp(' ');
            disp('File name conflict between:');
            disp(['source: ' srcFileName ' and']);
            disp(['dest:   ' dstFileName ]);
            Q = input('\nWould you like to add a string to the beginning of the destination file\nto resolve the conflict?\n(y/n): ','s');
            if strcmp(Q, 'y')
                prepend_string = input('\nSTRING: ', 's');
            end
            FileName = [prepend_string d(i).name];
            dstFileName = [HOMEIMAGES LABELMEFOLDER '/' FileName];
            Q = input('\nContinue using this string for future conflicts?\n(y/n): ','s');
            if ~strcmp(Q, 'y')
                clear prepend_string;
            end
        else
            FileName = [prepend_string d(i).name];
            dstFileName = [HOMEIMAGES LABELMEFOLDER '/' FileName];
        end
    end
    
    copyfile(srcFileName, dstFileName);
    
    I = imread(dstFileName);
    
    % read the annotation file
    ANN = imread([ANNFOLDER AnnName]);
    if length(size(ANN)) >2
        ANN = rgb2gray(ANN);
    end
    ANN = ANN > 0;   % convert to logical

    % create an annotation structure for this file
    clear A;
    A.annotation.filename = FileName;
    A.annotation.folder = LABELMEFOLDER;
    A.annotation.scendescription = scene_description;
    
    % extract annotated boundaries
    boundary = bwboundaries(ANN);
    
    count = 1;
    for b = 1:length(boundary)
        % extract the boundary points for object b
        c = boundary{b}(:,2);
        r = boundary{b}(:,1);
        [r,c] = reducem(r,c);
        
        if ~isempty(r)
    
            A.annotation.object(count).name = ann;
            A.annotation.object(count).deleted = 0;
            A.annotation.object(count).verified = 1;
            A.annotation.object(count).id = count;

            for q = 1:length(r)
                A.annotation.object(count).polygon.pt(q).x = num2str(c(q));
                A.annotation.object(count).polygon.pt(q).y = num2str(r(q));
            end
            count = count + 1;
        end
    end
    
    % option to visualize the first annotation to make sure it is correct
    if (i == 1) || (i == length(d))
        R = input(['\nWould you like to view image annotation ' num2str(i) ' to be imported?\n(y/n): '], 's');
        if strcmp(R, 'y')
            LMplot(A, 1, HOMEIMAGES);
            input('\nPress Enter to continue');
        end
    end
    
    file_root = FileName(1:max(regexp(FileName, '\.', 'start'))-1);
    XMLfilenm = [HOMEANNOTATIONS LABELMEFOLDER '/' file_root '.xml'];
    writeXML(XMLfilenm, A);
    disp(['wrote ' XMLfilenm]);
end
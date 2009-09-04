addpath([pwd, '/../../toolboxes/LabelMeToolbox/'], '-begin');   % append the path to kevin's toolbox
disp('--------------------------------------------------------------------------');
disp('  THIS TOOL IS FOR IMPORTING IMAGES CONTAINING A SINGLE ANNOTATION TO A ');
disp('  LABELME DATASET, SUCH AS CROPPED FACES.');
SOURCEFOLDER = input('To import images, first create a folder containing ONLY images with the \nsame annotation. Provide the path to the folder (e.g. /home/source/)\nPATH: ', 's');
HOMEIMAGES = input('\nProvide the path to your LabelMe HOMEIMAGESFOLDER \n(e.g. /home/user/LabelMe/Images/)\nPATH: ', 's');
HOMEANNOTATIONS = input('\nProvide the path to your LabelMe HOMEANNOTATIONS \n(e.g. /home/user/LabelMe/Annotations/)\nPATH: ', 's');
LABELMEFOLDER = input(['\nProvide a new or existing folder name in what the images will be\nstored: ' HOMEIMAGES 'FOLDER_NAME\nFOLDER NAME: '], 's');
scene_description = input('\nPlease enter keywords for the scene description \n(e.g. street urban city)\nKEYWORDS: ', 's');
ann = input('\nType the annotation associated with these images.\nANNOTATION: ', 's');
disp('--------------------------------------------------------------------------');

if ~strcmp(SOURCEFOLDER(length(SOURCEFOLDER)), '/')
    SOURCEFOLDER(length(SOURCEFOLDER)+1) = '/';
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

% create a list of the images in the source folder
d = dir([SOURCEFOLDER '*.png']);

clear prepend_string;
for i = 1:length(d)
    
    % copy the image to the destination folder
    FileName = d(i).name;
    srcFileName = [SOURCEFOLDER d(i).name];
    dstFileName = [HOMEIMAGES LABELMEFOLDER '/' FileName];
    
    
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
    
    % create an annotation structure for this file
    clear A;
    A.annotation.filename = FileName;
    A.annotation.folder = LABELMEFOLDER;
    A.annotation.scendescription = scene_description;
    
    % there is only 1 object per image corresponding to the entire image
    A.annotation.object.name = ann;
    A.annotation.object.deleted = 0;
    A.annotation.object.verified = 1;
    A.annotation.object.id = 1;
    
    A.annotation.object.polygon.pt(1).x = num2str(1);
    A.annotation.object.polygon.pt(1).y = num2str(1);

    A.annotation.object.polygon.pt(2).x = num2str(size(I,2));
    A.annotation.object.polygon.pt(2).y = num2str(1);
    
    A.annotation.object.polygon.pt(3).x = num2str(size(I,2));
    A.annotation.object.polygon.pt(3).y = num2str(size(I,1));
    
    A.annotation.object.polygon.pt(4).x = num2str(1);
    A.annotation.object.polygon.pt(4).y = num2str(size(I,1));
    
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
function [SET, DATASETS] = p_collect_data(DATASETS, LEARNERS, varargin)
%P_COLLECT_DATA organizes training images for viola-jones
%
%   TODO UPDATE!!!!
%   SET = p_collect_data(DATASETS, set_type) collects and processes 
%   training images specified by DATASETS and the set_type ('train' or
%   'validation'). Returns the structure SET containing the images
%   comprising the data set.
%
%   example:
%   ------------
%   TRAIN       = p_collect_data(DATASETS, LEARNERS, 'train');
%   VALIDATION  = p_collect_data(DATASETS, LEARNERS, 'validation')
%   TRAIN       = p_collect_data(DATASETS, LEARNERS, 'update', TRAIN);
%   VALIDATOION = p_collect_data(DATASETS, LEARNERS, 'update', VALIDATION);
%
%   Copyright 2009 Kevin Smith
%
%   See also P_RECOLLECT_DATA, P_TRAININGFILES


% process the input arugments to know if we need to collect initial data,
% update with new FP's, what the sample limits are, etc.
[NORM IMSIZE POS_LIM NEG_LIM UPDATE SET CASCADE DISPLAY set_type] = process_arguments(DATASETS, varargin);

% construct the LabelMe index if it has not been pre-loaded
if ~isfield(DATASETS, 'LabelMeIndex')
    % to save LabelMe indexing time, we check to see if we've already indexed
    DATASETS = load_labelme_index(DATASETS);

    %DATASETS.LabelMeIndex = LMdatabase(DATASETS.HOMEANNOTATIONS, DATASETS.LABELME_FOLDERS);
end

% populate SET source images matching the LabelMe index, if SET is not provided
if isempty(SET)
    SET.SourceImages = cell([1 length(DATASETS.LabelMeIndex)]);
    SET.SourceFiles = cell([1 length(DATASETS.LabelMeIndex)]);
    
    for i = 1:length(DATASETS.LabelMeIndex)
        filenm = [DATASETS.HOMEIMAGES '/' DATASETS.LabelMeIndex(i).annotation.folder '/' DATASETS.LabelMeIndex(i).annotation.filename];
        I = imread(filenm);
        I = preprocessImage(I, DATASETS);
        SET.SourceImages{i} = I;
        SET.SourceFiles{i} = [DATASETS.LabelMeIndex(i).annotation.folder '/' DATASETS.LabelMeIndex(i).annotation.filename];
    end
end

% fill SET with example images, locations, and special feature data (e.g. integral images)
if ~UPDATE
    % request initial positive and negative samples
    [SET, DATASETS] = request_data(SET, DATASETS.pos_query, POS_LIM, DATASETS, LEARNERS, 1, DISPLAY);            % pos
    [SET, DATASETS] = request_data(SET, DATASETS.neg_query, NEG_LIM, DATASETS, LEARNERS, -1, DISPLAY);           % neg
else
    % request replacment negative samples which generate FP's for all TN's
    [SET, DATASETS] = request_data(SET, DATASETS.neg_query, NEG_LIM, DATASETS, LEARNERS, -1, 'update', CASCADE); % neg update
end

% set a flag instructing if we should precompute feature responses (TRAIN)
if strcmp(set_type, 'train'); SET.precomputed = DATASETS.precomputed; end;
if strcmp(set_type, 'validation'); SET.precomputed = 0; end;





%%==========================================================================
% process_arguments(I, NORM) preprocesses images before loading them.
% functions for preprocessing the image can be defined and passed in
% p_settings.m if desired, as in the following example:
% flist = {@(x)imresize(x,[100 100]), @(x)imresize(x, [300 300])};

function I = preprocessImage(I, DATASETS)

% by default, we will convert images to grayscale
if length(size(I)) > 2
    I = rgb2gray(I);
end

% apply functions defined in p_settings.m to the image, if they exist
if isfield(DATASETS, 'flist')
    for f = 1:length(DATASETS.flist)
        I = DATASETS.flist{f}(I);
    end
end

%%==========================================================================
% process_arguments(DATASETS, varargin) determines if we are creating an
% initial data set, or if we have been passed a data set, finds the number
% of samples to add or replace, and the detector window size IMSIZE
function [NORM IMSIZE POS_LIM NEG_LIM UPDATE SET CASCADE DISPLAY set_type] = process_arguments(DATASETS, varargin)

vararg = varargin{1};

% define default parameters
POS_LIM = Inf; NEG_LIM = Inf;
set_type = 'train'; UPDATE = 0;
SET = []; CASCADE = []; DISPLAY = '';


for v = 1:length(vararg)
    a = vararg{v};
    if ischar(a)
        if strcmp(a, 'train')
            set_type = 'train';
        elseif strcmp(a, 'validation')
            set_type = 'validation';
        elseif strcmp(a, 'update')
            UPDATE = 1;
        elseif strcmp(a, 'display')
            DISPLAY = 'display';
        else
            error(['Error p_collect_data: unknown argument ' num2str(v+1) ' ' varargin{v}]);
        end
    elseif isstruct(a)
        if isfield(a, 'class')
            SET = a;
        elseif isfield(a, 'type')
            CASCADE = a;
        else
            error(['Error p_collect_data: unknown structure in argument ' num2str(v+1)]);
        end
    else
        error(['Error p_collect_data: unknown argument ' num2str(v+1)]);
    end
end


% get the detector window size IMSIZE
if isfield(DATASETS, 'IMSIZE')
    IMSIZE = DATASETS.IMSIZE;
else
    error('Error p_collect_data: You did not specify DATASETS.IMSIZE in p_settings.m');
end

% determine if we are supposed to normalize input images
if isfield(DATASETS, 'NORM')
    NORM = DATASETS.NORM;
else
    error('Error p_collect_data: You did not specify DATASETS.NORM in p_settings.m');
end

% get the sample limits for TRAIN data sets
if strcmp(set_type,'train')
    if isfield(DATASETS, 'TRAIN_POS')
        if ~isempty(DATASETS.TRAIN_POS)
            POS_LIM = DATASETS.TRAIN_POS;
        end
    end
    if isfield(DATASETS, 'TRAIN_NEG')
        if ~isempty(DATASETS.TRAIN_NEG)
            NEG_LIM = DATASETS.TRAIN_NEG;
        end
    end
end
    
% get the sample limits for VALIDATION data sets
if strcmp(set_type,'validation')
    if isfield(DATASETS, 'VALIDATION_POS')
        if ~isempty(DATASETS.VALIDATION_POS)
            POS_LIM = DATASETS.VALIDATION_POS;
        end
    end
    if isfield(DATASETS, 'VALIDATION_NEG')
        if ~isempty(DATASETS.VALIDATION_NEG)
            NEG_LIM = DATASETS.VALIDATION_NEG;
        end
    end
end
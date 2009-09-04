function [SET, DATASETS] = p_collect_data2(DATASETS, set_type)
%ADA_COLLECT_DATA organizes training images for viola-jones
%
%   SET = ada_collect_data(DATASETS, set_type) collects and processes 
%   training images specified by DATASETS and the set_type ('train' or
%   'validation'). Returns the structure SET containing the images
%   comprising the data set.
%
%   example:
%   ------------
%   TRAIN       = ada_collect_data(DATASETS, 'train');
%   VALIDATION  = ada_collect_data(DATASETS, 'validation')
%
%   Copyright 2009 Kevin Smith
%
%   See also P_RECOLLECT_DATA, P_TRAININGFILES


%% collect parameters from DATASETS 
% NORM = normalize images?, IMSIZE = training image size, POS_LIM = # positive examples, NEG_LIM = # of negative examples.

[NORM IMSIZE POS_LIM NEG_LIM] = collect_arguments(DATASETS, set_type);
count = 1;

%% collect POSITIVE (c = 1) and NEGATIVE (c = -1) example images into SET

for c = [-1 1]  % c = the postive and negative classes
    
    % collect the training image files into d, and initialize the data struct
    if c == 1
        query_string = DATASETS.labelme_pos_query;
        if ~isfield(DATASETS, 'LabelMeIndex')
            [data, DATASETS.LabelMeIndex] = p_request_data(query_string, POS_LIM, DATASETS, 'SIZE', IMSIZE);
        else
            [data, DATASETS.LabelMeIndex] = p_request_data(query_string, POS_LIM, DATASETS, DATASETS.LabelMeIndex, 'SIZE', IMSIZE);
        end
        
    else
        query_string = DATASETS.labelme_neg_query;
        if ~isfield(DATASETS, 'LabelMeIndex')
            [data, DATASETS.LabelMeIndex] = p_request_data(query_string, NEG_LIM, DATASETS, 'SIZE', IMSIZE);
        else
            [data, DATASETS.LabelMeIndex] = p_request_data(query_string, NEG_LIM, DATASETS, DATASETS.LabelMeIndex, 'SIZE', IMSIZE);
        end
    end
    
    % TODO: Handle this so we only save when we build the original index
    save DATASETS.mat DATASETS;

    % add each image file to SET, format it, normalize it, and compute features
    for i = 1:length(data)
       
        % store the image into SET
        SET.Images{count} = data{i};
        SET.IntImages{count} = integral_image(data{i})';
        if c == 1; SET.class(count) = 1; end
        if c == -1; SET.class(count) = -1; end

        count = count + 1;       
    end
    
    % set a flag instructing if we should precompute feature responses
    SET.precomputed = DATASETS.precomputed;
end



%==========================================================================
function [NORM IMSIZE POS_LIM NEG_LIM] = collect_arguments(DATASETS, set_type)

% define default parameters
NORM = 1; IMSIZE = [24 24];
POS_LIM = Inf;NEG_LIM = Inf;

% collect predefined arguments
if isfield(DATASETS, 'IMSIZE')
    if ~isempty(DATASETS.IMSIZE)
        IMSIZE = DATASETS.IMSIZE;
    end
end

if isfield(DATASETS, 'NORMALIZE')
    if ~isempty(DATASETS.NORMALIZE)
        NORM = DATASETS.NORMALIZE;
    end
end

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

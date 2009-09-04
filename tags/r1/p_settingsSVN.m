%% P_SETTINGS define settings/parameters 
%
%   Modify this file to set boosting parameters (learning schedule,
%   boosting type, etc), data parameters (data set, detector image size,
%   number of examples, etc), experiment & storage options, and weak
%   learner parameters.
%
%   See also P_TRAIN

%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%               Aurelien Lucchi     http://cvlab.epfl.ch/~lucchi/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

clear all;

% set paths we need for toolboxes
addpath([pwd, '/../toolboxes/kevin/']); 	% append the path to kevin's toolbox
addpath([pwd, '/../toolboxes/LabelMeToolbox/'], '-begin');   % append the path to kevin's toolbox
addpath([pwd, '/bin/']);                    % append the path to sub-functions
addpath([pwd, '/testing/']);                % append path to quality testing functions
addpath([pwd, '/images/']);                 % append path to test images subdirectory
addpath([pwd, '/../features/fragments/']);
addpath([pwd, '/../features/spedges/']);    % append the path to the ray's toolbox 

% seed the random number generator
s = RandStream.create('mt19937ar','seed',5489);  RandStream.setDefaultStream(s);  %rand('twister', 100);    % seed for Matlab 7.8 (?)

%-------------------------------------------------------------------------
% TRAINING PARAMETERS
%-------------------------------------------------------------------------

BOOST.type              = 'adaboost';   % select adaboost, realboost, dpboost
BOOST.function_handle   = @p_adaboost;  % function handle to the boosting script
BOOST.targetF           = 1e-5;         % target false positive rate for the entire cascade
BOOST.targetD           = .90;          % target detection rate for the entire cascade
BOOST.Nstages           = 1;           % number of cascade stages

%-------------------------------------------------------------------------
% EXPERIMENT & STORAGE INFORMATION
%-------------------------------------------------------------------------

EXPERIMENT.NAME             = ['Test' '_'];     % descriptive prefix string to identify experiment files
EXPERIMENT.datestr          = datestr(now, 'mmmddyyyy-HHMMSS');  %datestr(now, 'dd-mmm-yyyy-HH.MM.SS');
EXPERIMENT.computername     = 'calcifer';                         % computer experiment was run on
EXPERIMENT.cascade_filenm   = ['./results/' EXPERIMENT.NAME EXPERIMENT.datestr EXPERIMENT.computername '.mat'];   % filename to store the cascaded classifier
EXPERIMENT.log_filenm       = ['./logs/' EXPERIMENT.NAME EXPERIMENT.datestr EXPERIMENT.computername '.log'];   % filename to store the log file    

%-------------------------------------------------------------------------
% DATA SETS FOR TRAINING & VALIDATION 
%-------------------------------------------------------------------------

DATASETS.TRAIN_POS      = 1000;          % number of positive examples in the training set
DATASETS.TRAIN_NEG      = 1000;          % number of negative examples in the training set
DATASETS.VALIDATION_POS = 1000;          % number of positive examples in the validation set
DATASETS.VALIDATION_NEG = 1000;          % number of negative examples in the validation set

% parameters for updating the negative examples
DATASETS.IMSIZE             = [24 24];      % detector window size
DATASETS.precomputed        = 1;            % (1) speeds up training by precomputing feature responses (0) computes feature responses during training
DATASETS.delta              = 10;           % re-collection scan step size
DATASETS.NORM               = 0;            % normalize intensity? (1=FACES,NUCLEI,PERSONS, 0=MITO,CONTOURS)
DATASETS.labelme_pos_query  = 'face';
DATASETS.labelme_neg_query  = 'nonface';
DATASETS.LABELME_FOLDERS    = {'faces24x24', 'nonfaces24x24'};
% DATASETS.labelme_pos_query  = 'mitochondria+interior';
% DATASETS.labelme_neg_query  = '!mitochondria';
% DATASETS.LABELME_FOLDERS    = {'FIBSLICE'};
DATASETS.HOMEIMAGES         = '/osshare/Work/Data/LabelMe/Images';
DATASETS.HOMEANNOTATIONS    = '/osshare/Work/Data/LabelMe/Annotations';

%-------------------------------------------------------------------------
% WEAK LEARNERS
%-------------------------------------------------------------------------

% paramaeters of each type of weak learner are defined by a 2-letter prefix
% AB_ followed by parameters specific to each learner type which are parsed
% and interpreted by p_define_weak_learners.

LEARNERS.types = {'HA_x3_y3_u3_v3'};
%LEARNERS.types = {'IT', 'HA_x3_y3_u3_v3'};
%LEARNERS.types = {'FR_N_100_smin_17_smax_17_p_1'};
%LEARNERS.types = {'IT'};





% =========== OLD STUFF I DO NOT WANT TO GET RID OF YET ===================
%DATASETS.filelist = 'nuclei-rotated.txt';   DATASETS.scale_limits = [.6 2]; DATASETS.IMSIZE = [24 24];      
%DATASETS.filelist = 'faces.txt';            DATASETS.scale_limits = [.6 5]; DATASETS.IMSIZE = [24 24];
%DATASETS.filelist = 'mitochondria48.txt';   DATASETS.scale_limits = [2 9];  DATASETS.IMSIZE = [24 24];   
%DATASETS.filelist = 'mitochondria24.txt';   DATASETS.scale_limits = [2 9];  DATASETS.IMSIZE = [17 17];
%DATASETS.filelist = 'nuclei24.txt';         DATASETS.scale_limits = [13.3]; DATASETS.IMSIZE = [24 24];
%DATASETS.filelist = 'persons24x64.txt';     DATASETS.scale_limits = [1 5];  DATASETS.IMSIZE = [64 24];
%DATASETS.filelist = 'persons48x128.txt';    DATASETS.scale_limits = [1 5];  DATASETS.IMSIZE = [128 48];


%% load the data set we want to test on
% make sure that it is normalized/un-normalize if necessary!!!
path(path, [pwd '/..']);

LEARNERS = []; 

LEARNERS(length(LEARNERS)+1).feature_type   = 'intmean';
LEARNERS(length(LEARNERS)).IMSIZE        	= IMSIZE;

LEARNERS(length(LEARNERS)+1).feature_type 	= 'intvar';
LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;

LEARNERS(length(LEARNERS)+1).feature_type 	= 'haar';
LEARNERS(length(LEARNERS)).IMSIZE           = IMSIZE;
LEARNERS(length(LEARNERS)).shapes           = {'vert2', 'horz2', 'vert3', 'checker'};
LEARNERS(length(LEARNERS)).SCAN_Y_STEP      = 2;  % [6 persons, 1 all others]
LEARNERS(length(LEARNERS)).SCAN_X_STEP      = 2;  % [2 persons, 1 all others]

%load TEST_nuclei_un_norm.mat; N_POS = 1000; N_NEG = 70000;
%load TEST_nuclei_norm.mat;   N_POS = 1000; N_NEG = 70000;
load TEST_mito_un_norm.mat;  N_POS = 1200; N_NEG = 70000; 
%load TEST_faces_norm.mat;    N_POS = 1500; N_NEG = 100000;
%load TEST_persons_norm.mat;  N_POS = 1000; N_NEG = 20000;

%=============DEBUG==================
TEST.Images = TEST.Images(:,:,1:N_POS+N_NEG);
TEST.class = TEST.class(1:N_POS+N_NEG);
%====================================


%% load the CASCADE we will be evaluating and CUT it to size
%---------------------------------------------------------------------
%load HA-nucleirays1bMar0309-211918.mat;
%load SP-facescv8aMar052009-180951.mat;
%load COMBO-facesrays2Mar052009-175802.mat;

%load HA-nucleirays1bMar0309-211918.mat;
%load HO-nuclei-rays2Mar0309-211959.mat;
%load SP-nucleicv25Mar082009-053748.mat;
%load SP-nucleirays4bMar052009-195316.mat;
%load COMBO-nucleigbMar082009-051944.mat;   % ON GANDALF!

load HA-mito02-Mar-2009-00.37.16.mat;
%load HO-mito-02-Mar-2009-00.38.37.mat
%load SP-mitorays4aMar052009-175430.mat;
%load COMBO-mitorays4Mar072009-231100.mat;


nlearners = 1000;
CASCADE = ada_cut_cascade(CASCADE, nlearners);


% define a place to store the files
filenm    = [pwd '/' 'pre_HA_mito.mat'];

%% precompute the feature responses, and store them in FILES.test_filenm
ada_cascade_precom(TEST, CASCADE, LEARNERS, filenm);
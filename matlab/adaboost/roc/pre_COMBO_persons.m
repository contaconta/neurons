%% load the data set we want to test on
% make sure that it is normalized/un-normalize if necessary!!!
path(path, [pwd '/..']);

%load TEST_nuclei_un_norm.mat; N_POS = 1000; N_NEG = 70000;
%load TEST_nuclei_norm.mat;   N_POS = 1000; N_NEG = 70000;
%load TEST_mito_un_norm.mat;  N_POS = 1200; N_NEG = 70000; 
%load TEST_faces_norm.mat;    N_POS = 1500; N_NEG = 100000;
load TEST_persons_norm.mat;  N_POS = 1000; N_NEG = 20000;

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

%load HA-mito02-Mar-2009-00.37.16.mat;
%load HO-mito-02-Mar-2009-00.38.37.mat;
%load SP-mitorays4aMar052009-175430.mat;
%load COMBO-mitorays4Mar072009-231100.mat;

%load HO-personrays1bMar072009-235048.mat;
%oad SP-personcv30Mar082009-000047.mat;
load COMBO-persongaMar082009-143746.mat;

nlearners = 500;
CASCADE = ada_cut_cascade(CASCADE, nlearners);


% define a place to store the files
filenm    = [pwd '/' 'pre_COMBO_persons.mat'];

%% precompute the feature responses, and store them in FILES.test_filenm
ada_cascade_precom(TEST, CASCADE, LEARNERS, filenm);
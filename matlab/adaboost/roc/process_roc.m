%% load the CASCADE we will be evaluating and CUT it to size
%---------------------------------------------------------------------
%load HA-facescv8bMar052009-163516.mat;
%load SP-facescv8aMar052009-180951.mat;
%load COMBO-facesrays2Mar052009-175802.mat;

load HA-nucleirays1bMar0309-211918.mat;
%load HO-nuclei-rays2Mar0309-211959.mat;
%load SP-nucleicv25Mar082009-053748.mat;
%load SP-nucleirays4bMar052009-195316.mat;
%load COMBO-nucleigbMar082009-051944.mat;   % ON GANDALF!

%load HA-mito02-Mar-2009-00.37.16.mat;
%load HO-mito-02-Mar-2009-00.38.37.mat
%load SP-mitorays4aMar052009-175430.mat;
%load COMBO-mitorays4Mar072009-231100.mat;


nlearners = 408;
%nlearners = 500;
%nlearners = 1800;  %1800 NUCLEI, 1000 MITO, 500 FACES  (252 cvlabpc25)
%nlearners = 1000;
CASCADE = ada_cut_cascade(CASCADE, nlearners);

%gt = zeros([1 71200]);   % mito
%gt(1:1200) = 1;

%gt = zeros([1 101500]);   % faces
%gt(1:1500) = 1;

gt = zeros([1 71000]);    % nuclei
gt(1:1000) = 1;


%% load the precomputed features
%load pre_HA_faces.mat;
%load pre_SP_faces.mat;
%load pre_COMBO_faces.mat;

load pre_HA_nuclei.mat;
%load pre_HO_nuclei.mat;
%load pre_SP1_nuclei.mat;
%load pre_SP2_nuclei.mat;

%load pre_HA_mito.mat;
%load pre_HO_mito.mat;
%load pre_SP_mito.mat;


%% build the ROC curve!
%roc_HA_faces = plot_roc(CASCADE, A, gt);
%roc_SP_faces = plot_roc(CASCADE, A, gt);
%roc_COMBO_faces = plot_roc(CASCADE, A, gt);
%roc_HA_nuclei = plot_roc(CASCADE, A, gt);
roc_HA_nuclei408 = plot_roc(CASCADE, A, gt);
%roc_HO_nuclei = plot_roc(CASCADE, A, gt);
%roc_SP1_nuclei = plot_roc(CASCADE, A, gt);
%roc_SP2_nuclei = plot_roc(CASCADE, A, gt);
%roc_HA_mito = plot_roc(CASCADE, A, gt);
%roc_HO_mito = plot_roc(CASCADE, A, gt);
%roc_SP_mito = plot_roc(CASCADE, A, gt);

%% save the results!
%save roc_HA_faces.mat roc_HA_faces;
%save roc_SP_faces.mat roc_SP_faces;
%save roc_COMBO_faces.mat roc_COMBO_faces;

save roc_HA_nuclei408.mat roc_HA_nuclei408;
%save roc_HA_nuclei.mat roc_HA_nuclei;
%save roc_HO_nuclei.mat roc_HO_nuclei;
%save roc_SP1_nuclei.mat roc_SP1_nuclei;
%save roc_SP2_nuclei.mat roc_SP2_nuclei;

%save roc_HA_mito.mat roc_HA_mito;
%save roc_HO_mito.mat roc_HO_mito;
%save roc_SP_mito.mat roc_SP_mito;

grid on;  hold on;
xlabel('False Positive Rate');
ylabel('Correct Detection Rate');
%legend('Haar', 'HOG', 'Rays');
%set(gca, 'XTick', [0 .00002:.00002:.0002);


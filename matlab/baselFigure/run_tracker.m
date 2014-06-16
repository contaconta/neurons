if isempty( strfind(path, [pwd '/Common']) )
    addpath([pwd '/Common']);
end

if isempty( strfind(path, [pwd '/IO']) )
    addpath([pwd '/IO']);
end

if isempty( strfind(path, [pwd '/Geodesics']) )
    addpath([pwd '/Geodesics']);
end

run([pwd '/vlfeat-0.9.18/toolbox/vl_setup']);

if isempty( strfind(path, [pwd '/CellsDetection']) )
    addpath([pwd '/CellsDetection']);
end

if isempty( strfind(path, [pwd '/GreedyTracking']) )
    addpath([pwd '/GreedyTracking']);
end

if isempty( strfind(path, [pwd '/NeuritesDetection']) )
    addpath([pwd '/NeuritesDetection']);
end

if isempty( strfind(path, [pwd '/NeuritesTracking']) )
    addpath([pwd '/NeuritesTracking']);
end

if isempty( strfind(path, [pwd '/FeaturesExtraction']) )
    addpath([pwd '/FeaturesExtraction']);
end

if isempty( strfind(path, [pwd '/frangi_filter_version2a']) )
    addpath([pwd '/frangi_filter_version2a']);
end

if isempty( strfind(path, [pwd '/gaimc']) )
    addpath([pwd '/gaimc']);
end


%matlabpool
folder_n = '/home/ksmith/data/basel_figure/031/';
exp_num = 'mov031';
magnification = '10x';
Sample = 'PLATEX-GX-10X';
cellIDs = [1 5];
trkFigures(folder_n, exp_num, Sample, magnification, cellIDs);


% folder_n = '/home/ksmith/data/basel_figure/041/';
% exp_num = 'mov041';
% magnification = '10x';
% Sample = 'PLATEX-GX-10X';
% cellIDs = [1 2];
% trkFigures(folder_n, exp_num, Sample, magnification, cellIDs);
 
% folder_n = '/home/ksmith/data/basel_figure/132/';
% exp_num = 'mov132';
% magnification = '10x';
% Sample = 'PLATEX-GX-10X';
% cellIDs = [1 4 14];
% trkFigures(folder_n, exp_num, Sample, magnification, cellIDs);

% folder_n = '/home/ksmith/data/basel_figure/168/';
% exp_num = 'mov168';
% magnification = '10x';
% Sample = 'PLATEX-GX-10X';
% cellIDs = [4 5 6];
% trkFigures(folder_n, exp_num, Sample, magnification, cellIDs);

%matlabpool close
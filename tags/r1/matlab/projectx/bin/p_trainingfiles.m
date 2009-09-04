function files = p_trainingfiles(filenm, varargin)
%P_TRAININGFILES interfaces with a database text file
%   p_trainingfiles(filename, ...)
%   Used to create database text files or to parse and extract paths to
%   images from a database text file.
%
%   TO PARSE AND EXTRACT: Subsets of the dataset can be specified by 
%   including 'train', 'validation', 'update', or 'annotation' after the 
%   filename. Further specifying '+', '-', or 'both' will select the 
%   appropriate class. Example:
%       d = ada_trainingfiles('files.txt', 'train', '+' 500);
%   returns a list, d, containing 500 filenames of + class image files in 
%   the database specified by 'files.txt'. Other examples
%       d = ada_trainingfiles('files.txt', 'validation', '-' 1000);
%       d = ada_trainingfiles('files.txt', 'train', 'both' 500);
%   extracts 1000 negative examples from the validation set, and + and -
%   examples from the training set.
%
%   TO CREATE OR EXTEND: A new database text file can be created by placing
%   images files into a directory and specifying the path. Example:
%       ada_trainingfiles('files.txt', 'new', 'validation', '-','/path/to/images/');
%   creates a new database file, files.txt, with negative examples
%   belonging to the validation set. Training set examples can be added by:
%       ada_trainingfiles('files.txt', 'add', 'train', '+','/path/to/images/');
%   and 'update' examples used to repopulate the negative class can be
%   added by
%       ada_trainingfiles('files.txt', 'update', 'train','/path/to/images/');
%   Finally, filters can be added to specify specific files, for example:
%       ada_trainingfiles('files.txt', 'add', 'train', '+', path, {'*.jpg','*.png'});
%   adds all jpg and png files found in the path.
%
%   Copyright 2009 Kevin Smith
%
%   See also P_COLLECT_DATA

%% adding files or starting a new file
if strcmp(varargin{1}, 'add') || strcmp(varargin{1}, 'new')
    if strcmp(varargin{1}, 'add')
        fid = fopen(filenm, 'a');
    else
        fid = fopen(filenm, 'w');
    end
    
    s = varargin{2};        % the type of set, 'train', 'validation', or 'update'
    n = varargin{3};        % the class, '+' or '-'
    p = varargin{4};        % the path to the images
    if nargin == 6  
        filters = varargin{5};   % the filters to look for
    else
        filters = {'*.png', '*.jpg', '*.bmp', '*.tif', '*.tiff'};
    end
      
    d = [];
    for i = 1:length(filters)
        d = [d; dir([p filters{i}]);];
    end
    
    for i = 1:length(d)
       fprintf(fid,   '%5d\t%s\t%s\t%s\n', i, s, n, [p d(i).name] );
    end
end

%% requesting train images
if strcmp(varargin{1}, 'train')
    cl = varargin{2};       % the class is varargin2, '+', '-', or 'both'
    fid = fopen(filenm, 'r');
    if nargin == 4
        N = varargin{3};
    else 
        N = Inf;
    end
    
    C = textscan(fid, '%n%s%s%s');
    
    c_inds = find(strcmp(C{2},'train'));
    if strcmp(cl, 'both')
        n_inds = [ find(strcmp(C{3},'+')) ; find(strcmp(C{3},'-')) ];
    else
        n_inds = find(strcmp(C{3},cl));
    end
    inds = intersect(c_inds, n_inds);
    inds = inds(1: min([length(inds) N]));
    files = C{4}(inds);
end

%% requesting validation images
if strcmp(varargin{1}, 'validation')
    cl = varargin{2};       % the class is varargin2, '+', '-', or 'both'
    fid = fopen(filenm, 'r');
    if nargin == 4
        N = varargin{3};
    else 
        N = Inf;
    end
    
    C = textscan(fid, '%n%s%s%s');
    
    c_inds = find(strcmp(C{2},'validation'));
    if strcmp(cl, 'both')
        n_inds = [ find(strcmp(C{3},'+')) ; find(strcmp(C{3},'-')) ];
    else
        n_inds = find(strcmp(C{3},cl));
    end
    inds = intersect(c_inds, n_inds);
    inds = inds(1: min([length(inds) N]));
    files = C{4}(inds);
end

%% requesting update images
if strcmp(varargin{1}, 'update')
    cl = varargin{2};       % the class is varargin2, '+', '-', or 'both'
    fid = fopen(filenm, 'r');
    if nargin == 4
        N = varargin{3};
    else 
        N = Inf;
    end
    
    C = textscan(fid, '%n%s%s%s');
    
    c_inds = find(strcmp(C{2},'update'));
    if strcmp(cl, 'both')
        n_inds = [ find(strcmp(C{3},'+')) ; find(strcmp(C{3},'-')) ];
    else
        n_inds = find(strcmp(C{3},cl));
    end
    inds = intersect(c_inds, n_inds);
    inds = inds(1: min([length(inds) N]));
    files = C{4}(inds);
end

%% requesting annotation images
if strcmp(varargin{1}, 'annotation')
    cl = varargin{2};       % the class is varargin2, '+', '-', or 'both'
    fid = fopen(filenm, 'r');
    if nargin == 4
        N = varargin{3};
    else 
        N = Inf;
    end
    
    C = textscan(fid, '%n%s%s%s');
    
    c_inds = find(strcmp(C{2},'annotation'));
    if strcmp(cl, 'both')
        n_inds = [ find(strcmp(C{3},'+')) ; find(strcmp(C{3},'-')) ];
    else
        n_inds = find(strcmp(C{3},cl));
    end
    inds = intersect(c_inds, n_inds);
    inds = inds(1: min([length(inds) N]));
    files = C{4}(inds);
end

fclose(fid);
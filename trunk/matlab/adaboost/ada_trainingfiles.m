function files = ada_trainingfiles(filenm, varargin)
%
%
%  ada_trainingfiles('files.txt', 'add', 'train', '+', path, {'*.jpg', '*.png'});
%  ada_trainingfiles('files.txt', 'train', '+' 500);
%  ada_trainingfiles('files.txt', 'train', 'both' 500);
%  ada_trainingfiles('files.txt', 'new', 'validation', '-', path);
%  ada_trainingfiles('files.txt', 'add', 'update', path);
%  ada_trainingfiles('files.txt', 'add', 'annotation', path);


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
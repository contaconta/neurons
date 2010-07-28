function remove_outliers()


source = [pwd '/'];
%source = '/media/VERBATIM/Data/slice3noisycopy/';

d = dir([source '*.tif']);

% check if image stacks exist
if isempty(d)
    error('no image stacks found');
else
    T = length(d);
end

% initialize statistics
m = zeros(1,T); v = m; e = m;
names = cell(T,1);

% collect stats for each image stack
for t = 1:T
    
    fname = [source d(t).name];
    names{t} = fname;
    disp(['   checking ' fname]);
    V = readMultiPageTiff(fname);
    
    m(t) = median(V(:));
    v(t) = var(single(V(:)));
    e(t) = entropy(V(:));
end

% check for outliers
outliers = find(abs(m - median(m))  > 2*std(m));

if ~isempty(outliers)

    % find new indexes
    old = 1:T;
    new = setdiff(old, outliers);
    
    % delete the outliers
    for i = 1:length(outliers)
        disp(['removing outlier image stacks: ' names{outliers(i)} ]);
        delete(names{outliers(i)});
    end

    % rename/re-organize the remaining files
    disp('   reorganizing remaining image stacks');
    for i = 1:length(new)
        disp([ 'moving ' names{new(i)}  ' -> ' names{old(i)} ]);
        movefile(names{new{i}}, names{old{i}});
    end
end


%keyboard;






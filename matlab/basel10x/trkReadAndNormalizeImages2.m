function [stdSeq, maxSeq, minSeq, meanSeq, medianSeq] = trkReadAndNormalizeImages2(TMAX, folder)

I = cell(1,TMAX);

disp('');

list = dir(fullfile(folder, '*.TIF'));
names = {list.name};
sorted_filenames = sort_nat(names);

stdSeq = zeros(1, TMAX);
maxSeq = zeros(1, TMAX);
minSeq = zeros(1, TMAX);
meanSeq = zeros(1, TMAX);
medianSeq = zeros(1, TMAX);

for t = 1:TMAX
    if mod(t,10) == 0
        fprintf('|');
    end
    
    filename = [folder sorted_filenames{t}];
    I{t} = imread( filename );

    Ilist = double(I{t}(:));
    
    [h,x] = hist(Ilist,1000);
    hmax = max(h);
    h( h < .10 * hmax) = 0;
    minind = find(h > 0,1, 'first');
    maxind = find(h > 0,1, 'last');
    xmin = x(minind);
    xmax = x(maxind);
    
    maxSeq(t)    = max(Ilist);
    minSeq(t)    = min(Ilist);
    meanSeq(t)   = mean(Ilist);
    
    Ilist(Ilist < xmin) = [];
    Ilist(Ilist > xmax) = [];
   
    % center the image background on zero
   	originalMedian = median(Ilist);
    originalStd = std(Ilist);
    
    stdSeq(t)    = originalStd;
    medianSeq(t) = originalMedian;% median(Ilist);
    
end

fprintf('\n');
disp(['   loaded (' num2str(t) '/' num2str(TMAX) ') images from:  ' folder]);
disp('');


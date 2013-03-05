function [I_normalized, I_original] = trkReadImagesAndNormalize(TMAX, folder, IntensityAjustment)

I_normalized = cell(1,TMAX);
I_original   = cell(1,TMAX);

STD = IntensityAjustment.STD;

disp('');

list = dir(fullfile(folder, '*.TIF'));
names = {list.name};
sorted_filenames = sort_nat(names);

for t = 1:TMAX
    if mod(t,10) == 0
        fprintf('|');
    end
    filename = [folder sorted_filenames{t}];
    I_original{t}   = double( imread( filename ) );
    
    Ilist = double(I_original{t}(:));
    
    % normalize according to the background;
    [h,x] = hist(Ilist,1000);
    hmax = max(h);
    h( h < .10 * hmax) = 0;
    minind = find(h > 0,1, 'first');
    maxind = find(h > 0,1, 'last');
    xmin = x(minind);
    xmax = x(maxind);
    Ilist(Ilist < xmin) = [];
    Ilist(Ilist > xmax) = [];
   
    % center the image background on zero
   	originalMedian = median(Ilist);
    originalStd = std(Ilist);
    
    % correct the image
    I_normalized{t} = I_original{t} - originalMedian;    
    I_normalized{t} = I_original{t} * (STD / originalStd);

end

fprintf('\n');
disp(['   loaded (' num2str(t) '/' num2str(TMAX) ') images from:  ' folder]);
disp('');

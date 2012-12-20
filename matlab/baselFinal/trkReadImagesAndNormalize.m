function [I_normalized, I_original] = trkReadImagesAndNormalize(TMAX, folder)

I_normalized = cell(1,TMAX);
I_original   = cell(1,TMAX);

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
    I_normalized{t} = mat2gray( I_original{t} );
end

fprintf('\n');
disp(['   loaded (' num2str(t) '/' num2str(TMAX) ') images from:  ' folder]);
disp('');

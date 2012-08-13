function I = trkReadImagesAndNormalize(TMAX, folder)

I = cell(1,TMAX);

disp('');

list = dir(fullfile(folder, '*.TIF'));
names = {list.name};
sorted_filenames = sort_nat(names);

for t = 1:TMAX
    if mod(t,10) == 0
        fprintf('|');
    end
    filename = [folder sorted_filenames{t}];
    I{t} = mat2gray( double( imread( filename ) ) );
end

fprintf('\n');
disp(['   loaded (' num2str(t) '/' num2str(TMAX) ') images from:  ' folder]);
disp('');

% dir = '/media/data/steerableFilters2D/drive/d22/';
dir = '/media/data/steerableFilters2D/olivier/136/';

% imageName = 'd22_training_green.jpg';
imageName = '136.png';
oofName   = 'oof.jpg';
frangiName = 'frangi_2.jpg';
meijeeringName = 'meijeering.jpg';
sigma = 2.0;

oofDirectory   = '/home/ggonzale/workspace/steerableFilters2D/oof/main/';
addpath(oofDirectory);
matlab_directory   = '/home/ggonzale/workspace/steerableFilters2D/oof/matlab/';
addpath(matlab_directory);
mex_directory = '/home/ggonzale/workspace/steerableFilters2D/oof/mex/';
addpath(mex_directory);

d = oof([dir imageName]);
for i = 1:1:10
    save_float_image(d(:,:,i), [dir, 'oof_', num2str(i), '.jpg'])
end
% save_float_image(d, [dir oofName]);
% d = meijering([dir imageName], sigma);
% save_float_image(-d, [dir meijeeringName]);
% [d,a,b,c,d,e] = frangi([dir imageName], sigma);
% save_float_image(d, [dir frangiName]);


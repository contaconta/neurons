function motion2Dregistration(source)
%
%
%
%
%
%

%motion2Dfile = '/Users/feth/Documents/Work/CI_Softs/Motion2D-1.3.11/bin/Darwin/Motion2D';
motion2Dfile = '/osshare/Work/software/Motion2D-1.3.11/bin/Linux/Motion2D';


%% step 1: create a temp folder 
imgfolder = source;
d = dir([source '*.png']); T = length(d);
I = imread([source d(1).name]);


%% step 2: apply motion2D to the MIP image stacks
disp('   estimating motion parameters');
%cmd = [motion2Dfile ' -p ' source '0000%02d.png -s 1 -i ' num2str(length(d)-1) ' -r ' imgfolder 'motion_params.txt'];
outfolder = [imgfolder 'out/']; mkdir(outfolder);
cmd = [motion2Dfile ' -p ' source '0000%02d.png -s 1 -i ' num2str(length(d)-1)  ' -b ' outfolder 'out%02d.png ' ' -r ' imgfolder 'motion_params.txt'];
system('export LD_LIBRARY_PATH=/usr/X11/lib/');
system(cmd);  %disp('...Motion2D registration complete.');


%% step 3: load the transformation parameters from a file
[A Trans] = readparamsfile([imgfolder 'motion_params.txt']);


%% step 4: compute the flow fields
[X, Y] = meshgrid(0:size(I,2)-1, 0:size(I,1)-1);  %meshgrid(0:1030, 0:1030); %meshgrid(1:1031, 1:1031);
PosX = zeros(size(I,1)*size(I,2), T);  PosX(:, 1) = X(:); 
PosY = zeros(size(I,1)*size(I,2), T);  PosY(:, 1) = Y(:); 

disp('   computing flow');
for t = 2:T
    %disp(['   computing flow for t=' num2str(t)]);
    PosX_1 = PosX(:,t-1);
    PosY_1 = PosY(:,t-1);
    PosX(:, t) = Trans(t-1, 1)+ (A(t-1, 1, 1) * PosX_1 + A(t-1, 1, 2)*PosY_1);
    PosY(:, t) = Trans(t-1, 2)+ (A(t-1, 2, 1) * PosX_1 + A(t-1, 2, 2)*PosY_1);
%     PosX(:, t) = PosX(:,t-1) + Trans(t-1, 1);
%     PosY(:, t) = PosY(:,t-1) + Trans(t-1, 2);
end
PosX = reshape(PosX, [size(I,1) size(I,2) T]);
PosY = reshape(PosY, [size(I,1) size(I,2) T]);


%% step 5: apply the 2D transforms at each time step

% do not adjust the first file
tempdir = [source 'tmp/'];
mkdir(tempdir);
I = imread([source d(1).name]);
fname2 = [tempdir sprintf('%06d', 1) '.png'];
imwrite(I, fname2, 'PNG');

disp('   registering images');

% read the following files, apply the transform
for t = 2:T
    
    % read the tif
    fname = d(t).name;
    
    I = imread([source fname]);
    I = single(I);
    
    % interpolation
    xi = PosX(:,:,t);
    yi = PosY(:,:,t);    
    I2 = interp2(X, Y, I, xi, yi);
    I2 = uint8(I2);
    
    % overwrite with the registered file
    fname2 = [tempdir sprintf('%06d', t) '.png'];
    imwrite(I2, fname2, 'PNG');
    
    
end

filesToMovie(tempdir, 'sequenceREG.avi', 12, 'PNG');


keyboard;

%% Cleanup!

% remove the temporary directory
rmdir(tempdir, 's');

% remove the motion parameters file
delete([source 'motion_params.txt']);








function [A Trans] = readparamsfile(filename)


fid = fopen(filename);
for i = 1:43
    tline = fgetl(fid); %#ok<NASGU>
end
i = 1; params = zeros(1,17);
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    params(i,:) = str2num(tline); %#ok<ST2NM>
    i = i + 1;
end
params = params(:,2:9);

params = convert_to_absolutePositions(params);

Trans = params(:, 1:2);
A2 = params(:, 3:end);
A(:,1,1) = A2(:,1);
A(:,1,2) = A2(:,2);
A(:,2,1) = A2(:,3);
A(:,2,2) = A2(:,4);

% Trans = params(:, 3:4);
% A = [];
fclose (fid);



function out = convert_to_absolutePositions(params)


out = zeros(size(params, 1), size(params, 2)-2);

for k = 1:size(params, 1)
   out(k, 1) = params(k, 3) + params(k, 5)*(-params(k, 2)) + params(k, 6)*(-params(k, 1));
   out(k, 2) = params(k, 4) + params(k, 7)*(-params(k, 2)) + params(k, 8)*(-params(k, 1));
   out(k, 3) = params(k, 5)+1;
   out(k, 4) = params(k, 6);
   out(k, 5) = params(k, 7);
   out(k, 6) = params(k, 8)+1;
end



% function out = convert_to_absolutePositions(params)
% 
% 
% out = zeros(size(params, 1), size(params, 2)-2);
% 
% for k = 1:size(params, 1)
%    out(k, 1) = params(k, 3) + params(k, 5)*(-params(k, 1)) + params(k, 6)*(-params(k, 2));
%    out(k, 2) = params(k, 4) + params(k, 7)*(-params(k, 1)) + params(k, 8)*(-params(k, 2));
%    out(k, 3) = params(k, 5)+1;
%    out(k, 4) = params(k, 6);
%    out(k, 5) = params(k, 7);
%    out(k, 6) = params(k, 8)+1;
% end





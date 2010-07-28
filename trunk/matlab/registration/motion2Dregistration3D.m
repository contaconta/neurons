function motion2Dregistration3D(source)
%
%
%
%
%
%

%motion2Dfile = '/Users/feth/Documents/Work/CV_Softs/Motion2D-1.3.11/bin/Darwin/Motion2D';
motion2Dfile = '/osshare/Work/software/Motion2D-1.3.11/bin/Linux/Motion2D';


%% step 1: read in the series of image stacks, create a temp folder to hold the MIPs
mipfolder = [source 'xyMIP/'];    d = dir([source '*.tif']); T = length(d);
if isempty(d)
    error('no image stack files found!');
end
mkdir(mipfolder);

disp('   reading image stacks');
for i = 1:T
    fname = d(i).name; %disp(['...computing MIP for ' fname]);

    % read the tiff, clean it, write MIPS to a folder
    V = readMultiPageTiff([source fname]);    
    mip = max(V, [],3);                     % compute the MIP    
    filename = [mipfolder sprintf('%06d', i) '.png'];
    imwrite(mip, filename, 'PNG');          % write the MIP  
end


%% step 2: apply motion2D to the MIP image stacks
disp('   estimating motion parameters');
cmd = [motion2Dfile ' -p ' mipfolder '0000%02d.png -s 1 -i ' num2str(length(d)-1) ' -r ' mipfolder 'motion_params.txt'];
system('export LD_LIBRARY_PATH=/usr/X11/lib/');
system(cmd);


%% step 3: load the transformation parameters from a file
[A Trans] = readparamsfile([mipfolder 'motion_params.txt']);


%% step 4: compute the flow fields
[X, Y] = meshgrid(0:size(V,2)-1, 0:size(V,1)-1);  %meshgrid(0:1030, 0:1030); %meshgrid(1:1031, 1:1031);
PosX = zeros(size(V,1)*size(V,2), T);  PosX(:, 1) = X(:); 
PosY = zeros(size(V,1)*size(V,2), T);  PosY(:, 1) = Y(:); 

disp('   computing flow');
for t = 2:T
    PosX_1 = PosX(:,t-1);
    PosY_1 = PosY(:,t-1);
    PosX(:, t) = Trans(t-1, 1)+ (A(t-1, 1, 1) * PosX_1 + A(t-1, 1, 2)*PosY_1);
    PosY(:, t) = Trans(t-1, 2)+ (A(t-1, 2, 1) * PosX_1 + A(t-1, 2, 2)*PosY_1);
end
PosX = reshape(PosX, [size(V,1) size(V,2) T]);
PosY = reshape(PosY, [size(V,1) size(V,2) T]);


%% step 5: apply the 2D transforms at each time step

%---------------TEMPORARY------------------
mkdir([mipfolder '/out/']);
%------------------------------------------

disp('   registering images');
for t = 1:T
    
    % read the tif
    fname = d(t).name;
    disp(['...registered (' num2str(t) '/' num2str(T) ')']);
    V = readMultiPageTiff([source fname]);
    V = single(V);
    
    % interpolation
    xi = PosX(:,:,t);
    yi = PosY(:,:,t);    
    for z = 1:size(V,3)
      	V(:,:,z) = interp2(X, Y, V(:,:,z), xi,yi);
    end
    
    % overwrite with the registered file
    writeMultiPageTiff(uint8(V), [source fname]);
    
    %---------------TEMPORARY------------------
    % write a registered MIP so we can see if we've done well
    filename = [mipfolder '/out/' sprintf('%06d', t) '.png'];
    imwrite( max(uint8(V), [ ], 3), filename, 'PNG');
    %------------------------------------------
end


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








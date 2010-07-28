function motion2Dregistration3Dz(source)
%
%
%
%
%
%

%motion2Dfile = '/Users/feth/Documents/Work/CV_Softs/Motion2D-1.3.11/bin/Darwin/Motion2D';
motion2Dfile = '/osshare/Work/software/Motion2D-1.3.11/bin/Linux/Motion2D';


%% step 1: read in the series of image stacks, create a temp folder to hold the MIPs
mipfolderXY = [source 'xyMIP/'];    d = dir([source '*.tif']); T = length(d);
if isempty(d)
    error('no image stack files found!');
end
mkdir(mipfolderXY);

mipfolderXZ = [source 'xzMIP/'];
mkdir(mipfolderXZ);

% disp('   reading image stacks');
% for i = 1:T
%     fname = d(i).name; %disp(['...computing MIP for ' fname]);
% 
%     % read the tiff, clean it, write MIPS to a folder
%     V = readMultiPageTiff([source fname]);    
%     mipXY = max(V, [],3);                     % compute the MIP    
%     mipXZ = squeeze(max(V, [],1));                     % compute the MIP    
%     filenameXY = [mipfolderXY sprintf('%06d', i) '.png'];
%     filenameXZ = [mipfolderXZ sprintf('%06d', i) '.png'];
%     imwrite(mipXY, filenameXY, 'PNG');          % write the MIP  
%     imwrite(mipXZ, filenameXZ, 'PNG');          % write the MIP  
% end
% Vsize = size(V);
% clear V;
% 
% %% step 2: apply motion2D to the MIP image stacks
% disp('   estimating motion parameters');
% cmd = [motion2Dfile ' -p ' mipfolderXY '0000%02d.png -s 1 -i ' num2str(length(d)-1) ' -r ' mipfolderXY 'motion_params.txt'];
% system('export LD_LIBRARY_PATH=/usr/X11/lib/');
% system(cmd);
% cmd = [motion2Dfile ' -p ' mipfolderXZ '0000%02d.png -s 1 -i ' num2str(length(d)-1) ' -r ' mipfolderXZ 'motion_params.txt'];
% system(cmd);

Vsize = [1031 1031 101];

%% step 3: load the transformation parameters from a file
[AXY TransXY] = readparamsfile([mipfolderXY 'motion_params.txt']);
[AXZ TransXZ] = readparamsfile([mipfolderXZ 'motion_params.txt']);


%% step 4: compute the flow fields
[X, Y] = meshgrid(single(0:Vsize(2)-1), single(0:Vsize(1)-1));  %meshgrid(0:1030, 0:1030); %meshgrid(1:1031, 1:1031);
[XX, Z] = meshgrid(single(0:Vsize(2)-1), single(0:Vsize(3)-1));  %meshgrid(0:1030, 0:1030); %meshgrid(1:1031, 1:1031);
PosX = zeros(Vsize(1)*Vsize(2), T, 'single');  PosX(:, 1) = X(:); 
PosY = zeros(Vsize(1)*Vsize(2), T, 'single');  PosY(:, 1) = Y(:);
PosZ = zeros(Vsize(2)*Vsize(3), T, 'single');  PosZ(:, 1) = Z(:);
PosXX = zeros(Vsize(2)*Vsize(3), T, 'single');  PosXX(:, 1) = XX(:);
%PosX = repmat(PosX, [1, 1, Vsize( 3)]);
%PosY = repmat(PosY, [1, 1, Vsize( 3)]);
%%
disp('   computing flow');
for t = 2:T
    PosX_1 = PosX(:,t-1);
    PosY_1 = PosY(:,t-1);
    PosZ_1 = PosZ(:,t-1);
    PosXX_1 = PosXX(:,t-1);
    PosX(:, t) = TransXY(t-1, 1)+ (AXY(t-1, 1, 1) * PosX_1 + AXY(t-1, 1, 2)*PosY_1);
    PosY(:, t) = TransXY(t-1, 2)+ (AXY(t-1, 2, 1) * PosX_1 + AXY(t-1, 2, 2)*PosY_1);
%     PosXX(:, t)= TransXZ(t-1, 1)+ (AXZ(t-1, 1, 1) * PosXX_1 + AXZ(t-1, 1, 2)*PosZ_1);
%     PosZ(:, t) = TransXZ(t-1, 2)+ (AXZ(t-1, 2, 1) * PosXX_1 + AXZ(t-1, 2, 2)*PosZ_1);
    PosZ(:, t)= TransXZ(t-1, 1)+ (AXZ(t-1, 1, 1) * PosZ_1 + AXZ(t-1, 1, 2)*PosXX_1);
    PosXX(:, t) = TransXZ(t-1, 2)+ (AXZ(t-1, 2, 1) * PosZ_1 + AXZ(t-1, 2, 2)*PosXX_1);
end
PosX = reshape(PosX, [Vsize(1) Vsize(2) T]);
PosY = reshape(PosY, [Vsize(1) Vsize(2) T]);
PosZ = reshape(PosZ, [Vsize(3) Vsize(2) T]);
%keyboard;

%PosZ = reshape(PosZ, [Vsize(2) Vsize(3) T]);
clear PosX_1 PosY_1 PosZ_1 PosXX_1;
clear X Y Z XX;
clear PosXX;

%% step 5: apply the 2D transforms at each time step

%---------------TEMPORARY------------------
mkdir([mipfolderXY '/out/']);
%------------------------------------------

%[X Y Z] = meshgrid(single(0:Vsize(2)-1), single(0:Vsize(1)-1), single(0:Vsize(3)-1));

disp('   registering images');
for t = 1:T
    
    % read the tif
    fname = d(t).name;
    disp(['...registering (' num2str(t) '/' num2str(T) ')']);
    V = readMultiPageTiff([source fname]);
    V = single(V);
    
    % interpolation
    xi = single(PosX(:,:,t));
    yi = single(PosY(:,:,t));  
    zi = single(PosZ(:,:,t));
    % reshape
    xi = repmat(xi, [1, 1, Vsize( 3)]) + 1;  % +1 to not specify X,Y,Z for interp3
    yi = repmat(yi, [1, 1, Vsize( 3)]) + 1;
    %zi = repmat(zi, [Vsize(1), 1, 1]) + 1;
    Zi = zeros(Vsize, 'single');
    for y = 1:Vsize(2)
        Zi(:, y, :) = zi;
    end
    zi = Zi + 1;
    clear Zi;
    
    
    %keyboard;
    
      	%V(:,:,z) = interp2(X, Y, V(:,:,z), xi,yi);
        %V = interp3(X, Y, Z, V, xi,yi,zi);

        V = interp3(V, xi,yi,zi);
   
    
    % overwrite with the registered file
    writeMultiPageTiff(uint8(V), [source fname]);
    
    %---------------TEMPORARY------------------
    % write a registered MIP so we can see if we've done well
    filename = [mipfolderXY '/out/' sprintf('%06d', t) '.png'];
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








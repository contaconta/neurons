%% find the intensity limits of the image sequences, [rmin rmax] [gmin gmax]
function [rmin rmax gmin gmax R G mv] = readImagesAndGetIntensityLimits(TMAX, Rfolder, Rfiles, Gfolder, Gfiles)

rmax = 0;  rmin = 255;  gmax = 0;  gmin = 2^16;
R = cell(1,TMAX);
G = R;
mv = R;

disp('');
for t = 1:TMAX
    if mod(t,10) == 0
        %disp(['   t = ' num2str(t) '/' num2str(TMAX)]);
        fprintf('|');
    end
    
    R{t} = imread([Rfolder Rfiles(t).name]);
    rmax = max(rmax, max(R{t}(:)));
    rmin = min(rmin, min(R{t}(:)));
    G{t} = imread([Gfolder Gfiles(t).name]);
    gmax = max(gmax, max(G{t}(:)));
    gmin = min(gmin, min(G{t}(:)));
    
    if t == 1
        lims = stretchlim(G{t});
    end
    G8bits = trkTo8Bits(G{t}, lims);
    
    % make an output image
    Ir = mat2gray(G8bits);
    I(:,:,1) = Ir;
    I(:,:,2) = Ir;
    I(:,:,3) = Ir;
    
    mv{t} = I;
end
fprintf('\n');
disp(['   loaded (' num2str(t) '/' num2str(TMAX) ') images from:  ' Gfolder]);
disp('');


%% convert 16-bit image to 8-bit image
function J = trkTo8Bits(I, lims)

J = imadjust(I, lims, []);
J = uint8(J/2^8);
%% detect filaments
function F = detectFilaments(f, gmin, TMAX, MIN_FILAMENT_SIZE, SMASK)

%FRANGI_THRESH = double(gmin) * 1e-11;
FRANGI_THRESH = double(gmin) * 4e-11;

MIN_SKEL_SIZE = 12;
%MIN_FILAMENT_DISTANCE = 45;
MIN_FILAMENT_DISTANCE = 30;

BORDER = 5;
B = zeros(size(f{1}));
B = B > Inf;
B(1:end,1:BORDER) = 1; 
B(1:end,end-BORDER+1:end) = 1;
B(1:BORDER,1:end) = 1; 
B(end-BORDER+1:end,1:end) = 1;


for t = 1:TMAX
    F{t} = f{t} > FRANGI_THRESH;
    F{t} = bwareaopen(F{t}, MIN_FILAMENT_SIZE, 8);
    F{t} = bwmorph(F{t}, 'skel', Inf);
    F{t}(SMASK{t}) = 0;
    F{t}(B) = 0;
   	F{t} = bwareaopen(F{t}, MIN_SKEL_SIZE, 8);
    
    D = bwdist(SMASK{t});
    L=bwlabel(F{t});
    for i = 1:max(max(L))
        M = (L == i);
        mindist = min(D(M));
        
        if mindist > MIN_FILAMENT_DISTANCE
            F{t}(M) = 0;
        end
    end
end

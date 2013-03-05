%% detect endpoints and branch points
function [EndP BranchP] = detectEndBranch(F, TMAX, SMASK)

for t = 1:TMAX
    SM = SMASK{t};
    SM = bwmorph(SM, 'dilate');
    EndP{t} = bwmorph(F{t}, 'endpoints');
    BranchP{t} = bwmorph(F{t}, 'branchpoints');
    EndP{t}(SM) = 0;
    BranchP{t}(SM) = 0;
end
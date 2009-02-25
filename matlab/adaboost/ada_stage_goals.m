% Ftarget   - overall false positive rate target
% Dtarget   - overall target detection rate
% fmax      - default goal false positive rate for cascade stage
% dmin      - default minimum detection rate for cascade stage
clear GOALS;

NStages = 13;

GOALS(1).dmin = .99;
GOALS(1).fmax = .5;
%---------------------------------------------------
GOALS(2).dmin = .99;
GOALS(2).fmax = .5;
%---------------------------------------------------
GOALS(3).dmin = .99;
GOALS(3).fmax =  .45;
%---------------------------------------------------
GOALS(4).dmin = .991;
GOALS(4).fmax =  .45;
%---------------------------------------------------
GOALS(5).dmin = .992;
GOALS(5).fmax =  .4;
%---------------------------------------------------
GOALS(6).dmin = .993;
GOALS(6).fmax =  .4;
%---------------------------------------------------

S = 7;  Dcurrent = prod([GOALS(:).dmin]); Fcurrent = prod([GOALS(:).fmax]);
for s = S:NStages;
    GOALS(s).dmin = (Dtarget/Dcurrent)^(1/abs(NStages-S+1));
    GOALS(s).fmax = (Ftarget/Fcurrent)^(1/abs(NStages-S+1));
end

clear S NStages;

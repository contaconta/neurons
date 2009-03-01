% Ftarget   - overall false positive rate target
% Dtarget   - overall target detection rate
% fmax      - default goal false positive rate for cascade stage
% dmin      - default minimum detection rate for cascade stage
clear GOALS;

NStages = 1;

GOALS(1).dmin = .90;
GOALS(1).fmax = .00005;


clear S NStages;

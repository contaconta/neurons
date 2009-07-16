function BOOST = p_stage_goals(BOOST)
%P_STAGE_GOALS defines the performance goals of each cascade stage
%
%   GOALS = p_stage_goals(targetF, targetD, NStages) returns a structure
%   GOALS containing the cascade stage false positive performance goal
%   (fmax) and the cascade stage minimum detection goal (dmin) for a
%   schedule of NStages.
%
%   targetF   - overall false positive rate target
%   targetD   - overall target detection rate
%   fmax      - max false positive rate for cascade stage
%   dmin      - minimum detection rate for cascade stage

%   Copyright © 2009 Computer Vision Lab, 
%   École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.
%   All rights reserved.
%
%   Authors:    Kevin Smith         http://cvlab.epfl.ch/~ksmith/
%               Aurelien Lucchi     http://cvlab.epfl.ch/~lucchi/
%
%   This program is free software; you can redistribute it and/or modify it 
%   under the terms of the GNU General Public License version 2 (or higher) 
%   as published by the Free Software Foundation.
%                                                                     
% 	This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

% we begin by hand-tuning the first few stages to have a permissive FP rate

NStages = BOOST.Nstages;
targetF = BOOST.targetF;
targetD = BOOST.targetD;

BOOST.goals(1).dmin = .90;
BOOST.goals(1).fmax = 1e-5;

% BOOST.goals(1).dmin = .99;
% BOOST.goals(1).fmax = .5;
% %---------------------------------------------------
% BOOST.goals(2).dmin = .99;
% BOOST.goals(2).fmax = .5;
% %---------------------------------------------------
% BOOST.goals(3).dmin = .99;
% BOOST.goals(3).fmax =  .45;
% %---------------------------------------------------
% BOOST.goals(4).dmin = .991;
% BOOST.goals(4).fmax =  .45;
% %---------------------------------------------------
% BOOST.goals(5).dmin = .992;
% BOOST.goals(5).fmax =  .4;
% %---------------------------------------------------
% BOOST.goals(6).dmin = .993;
% BOOST.goals(6).fmax =  .4;
% %---------------------------------------------------
% 
% S = length(BOOST.goals); Dcurrent = prod([BOOST.goals(:).dmin]); Fcurrent = prod([BOOST.goals(:).fmax]);
% 
% % now automatically determine the goals for the remaining stages
% for s = S:NStages;
%     BOOST.goals(s).dmin = (targetD/Dcurrent)^(1/abs(NStages-S+1));
%     BOOST.goals(s).fmax = (targetF/Fcurrent)^(1/abs(NStages-S+1));
% end


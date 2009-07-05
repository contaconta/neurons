function [CASCADE, Fi, Di]  = p_cascade_select_threshold(CASCADE, i, VALIDATION, dmin)
%% P_CASCADE_SELECT_THRESHOLD adjusts cascade threshold to meet dmin 
%
%   [CASCADE, Fi, Di]  = p_cascade_select_threshold(CASCADE, i, DATA, dmin)
%   adjusts the cascade threshold so that the minimum detection rate is met
%   when the cascaded classifier is applied to the validation data. Returns
%   the adjusted CASCADE along with the overall detection rate Di and the
%   overall false positive rate Fi.
%
%   See also P_TRAIN

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

%% TODO: CHECK OVER AND REWRITE FOR NEW CLASSIFICATION FUNCTION

CASCADE(i).threshold = 2;           % set the initial sensitivity threshold
gt_all = [VALIDATION.class]';           % the validation ground truth 
C = zeros(size(gt_all));                % init a vector for our cascade results

% COLLECT THE MISCLASSIFICATIONS THAT MAKE IT THROUGH CASCADE(1:i-1)
if i > 1
    C = p_classify_cascade(CASCADE(1:i-1), VALIDATION);
    
    [TPs FPs] = rocstats(C, gt_all, 'TPlist', 'FPlist');  
    PASSTHROUGHS = [TPs; FPs];
    gt = VALIDATION.class(PASSTHROUGHS);
else
    gt = gt_all;
    PASSTHROUGHS = 1:length(VALIDATION.class);
    FPs = zeros(size(gt)); 
end

% TODO: need to handle the case when PASSTHROUGHS is empty !!
if isempty(PASSTHROUGHS)
    disp('The previous stage of the cascade did not produce any positive classifications, something is wrong!');
    keyboard;
end



Fi = prod([CASCADE(:).fi]);
Di = prod([CASCADE(:).di]);

if i == 1; Dlast = 1; else Dlast = prod([CASCADE(1:i-1).di]); end

tempTHRESH = [];  tempDi =[];

% search for the correct classifier threshold using a binary search
low = 0;  high = 2;  accuracy = .0001;  iterations = 0;  MAX_ITERATIONS = 20; STOP_ITERATIONS = 12;
while (low <= high) && (iterations < MAX_ITERATIONS)
    iterations = iterations + 1;
    THRESH = (low + high) /2;
    CASCADE(i).threshold = THRESH;

    %C = dummy_classify_set(CASCADE, VALIDATION);
    C = p_classify_cascade(CASCADE, VALIDATION);
    C = C(PASSTHROUGHS);
    
    [CASCADE(i).di CASCADE(i).fi fps tps] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist', 'TPlist'); 
    Fi = prod([CASCADE(:).fi]);
    Di = prod([CASCADE(:).di]);
    %disp(['low = ' num2str(low) '  high = ' num2str(high) '  THRESH = ' num2str(THRESH) '  Di = ' num2str(Di) '  Fi = ' num2str(Fi) '  DiTARGET = ' num2str(dmin * Dlast) ]);
    if (Di - dmin * Dlast) < 0 
        high = THRESH;
    elseif (Di - dmin * Dlast ) > accuracy
        low = THRESH;
    else
        %disp(['found TARGET THRESH = ' num2str(THRESH) ]);
        break
    end
    
    if ( iterations > STOP_ITERATIONS ) && (Di > dmin*Dlast)
        %disp(['stopped after ' num2str(STOP_ITERATIONS) ' iterations.']);
        break;
    end
    
    tempTHRESH(length(tempTHRESH)+1) = THRESH;
    tempDi(length(tempDi)+1) = Di;
end    

% display the results of our search for a suitable threshold
%disp(['Di=' num2str(Di) ', Fi=' num2str(Fi) ', #FPs->Stage' num2str(i) ' = ' num2str(length(FPs)) '. CASCADE applied to VALIDATION set.']);               
%disp(['di=' num2str(CASCADE(i).di) ', fi=' num2str(CASCADE(i).fi) ', #fp = ' num2str(length(fps)) '. Stage ' num2str(i) ' applied to VALIDATION (selected threshold ' num2str(CASCADE(i).threshold) ').' ]);               



S1 = sprintf('     Di=%5.4g (%d/%d)\tFi=%5.4g (%d/%d)\tCASCADE -> VALIDATION SET', Di, length(tps), length(find(gt_all == 1)), Fi, length(fps), length(find(gt_all == -1)) );
S2 = sprintf('     di=%5.4g (%d/%d)\tfi=%5.4g (%d/%d)\tSTAGE %d -> VALIDATION SET', CASCADE(i).di, length(tps), length(find(gt == 1)), CASCADE(i).fi, length(fps), length(find(gt == -1)), i);

S3 = ['                      selected CASCADE.threshold = ' sprintf('%0.6g', CASCADE(i).threshold) ];

%disp(['  Detection             False Positive         selected CASCADE.threshold = ' sprintf('%0.6g', CASCADE(i).threshold)])   
disp(S3);
disp(S2);
disp(S1);

   
%keyboard;

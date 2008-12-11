function [CASCADE, Fi, Di]  = ada_cascade_select_threshold(CASCADE, i, VALIDATION, dmin)
% search for a threshold for the current classifier which gives a
% satifactory detection rate


CASCADE(i).threshold = 2;           % set the initial sensitivity threshold
%gt = [VALIDATION(:).class]';        % the validation ground truth 
gt = [VALIDATION.class]';        % the validation ground truth 
C = zeros(size(gt));                % init a vector for our cascade results


% COLLECT THE MISCLASSIFICATIONS THAT MAKE IT THROUGH CASCADE(1:i-1)
if i > 1
    for j=1:length(VALIDATION.class); 
        C(j) = ada_classify_cascade(CASCADE(1:i-1), VALIDATION, j);
    end   
    [TPs FPs] = rocstats(C, gt, 'TPlist', 'FPlist');  
    PASSTHROUGHS = VALIDATION([TPs; FPs]);
    gt = [VALIDATION([TPs; FPs]).class]';
    C = zeros(size(gt));            % re-init a vector for our cascade results
else
    PASSTHROUGHS = VALIDATION;
    FPs = zeros(length(gt));
end

% TODO: need to handle the case when PASSTHROUGHS is empty !!


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
%     for j=1:length(PASSTHROUGHS); 
%         C(j) = ada_classify_strong(CASCADE(i).CLASSIFIER, PASSTHROUGHS, j, THRESH);
%     end  
    C = ada_classify_set(CASCADE(i), VALIDATION, THRESH);
    
    [CASCADE(i).di CASCADE(i).fi fps] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist'); 
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
disp(['results on VALIDATION data for CASCADE: Di=' num2str(Di) ', Fi=' num2str(Fi) ', #FPs = ' num2str(length(FPs)) ]);               
disp(['results on VALIDATION data for stage ' num2str(i) ' (threshold = ' num2str(CASCADE(i).threshold) '): di=' num2str(CASCADE(i).di) ', fi=' num2str(CASCADE(i).fi) ', #fps = ' num2str(length(fps)) ]);               


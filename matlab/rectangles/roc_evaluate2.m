function [TP FP] = roc_evaluate2(VALS, L)
%
%   VALS = Strong classifier responses 
%   L = class labels of the test set
%


[VSORT, inds] = sort(VALS);
LSORT = L(inds);

current_thresh = VSORT(1);
TP = sum( (LSORT==1) .* (VSORT>=current_thresh) );
FP = sum( (LSORT==-1).* (VSORT>=current_thresh)  );

last_thresh = VSORT(1);

%disp('...looping to compute the ROC'); 
tic;

c = 1;

for i = 2:length(VSORT)

    if (LSORT(i-1) == -1) && (LSORT(i) == 1) && (VSORT(i) > last_thresh)
        c = c+1;
        
        current_thresh = VSORT(i);
        
        TP(c) = sum( (LSORT==1) .* (VSORT >= current_thresh) );
        FP(c) = sum( (LSORT==-1).* (VSORT >= current_thresh) );

        last_thresh = current_thresh;
    end
end
to=toc;  %disp(['   Elapsed time ' num2str(to) ' seconds.']);


%keyboard;
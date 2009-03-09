function ROCDATA = plot_roc(CASCADE, PREMAT, gt, varargin)

T_LIMS = [0 1.5];
T_STEP = .01;

if nargin > 3
    T_LIMS = varargin{1};
end

if nargin > 4
    % we want to specify the number of learners!
    nlearners = varargin{2};
    %CUT THE CASCADE
end


LAST_STAGE = length(CASCADE);

count = 1;
disp('computing ROC for final stage');
for thresh = T_LIMS(1): T_STEP :T_LIMS(2);

    CASCADE(LAST_STAGE).threshold = thresh;
    C = ada_test_classify_set(CASCADE, PREMAT);
    
    [tpr fpr FPs TP] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist', 'TP');
    %disp([num2str(learners) ' learners, Di=' num2str(tpr) ', Fi=' num2str(fpr) ', #FP = ' num2str(length(FPs)) '.  CASCADE stage ' num2str(i) ' learner ' num2str(ti) ' applied to TEST set.'  ]);               
        
    ROCDATA(count,:) = [tpr fpr length(FPs) TP];
    count = count + 1;
end


plot(ROCDATA(:,2), ROCDATA(:,1));













% CASCADE(LAST_STAGE).thresh = 0;
% 
% CASCADE = CASCADE(1:LAST_STAGE-1);
% 
% while length(CASCADE) > 1
%    for s = LAST_STAGE-1:-1:1 
%         disp(['computing ROC for stage ' num2str(s)]);
%         ORIGINAL_THRESH = CASCADE(s).threshold;
%        
%         for thresh = T_LIMS(1):T_STEP:ORIGINAL_THRESH
%             CASCADE(s).threshold = thresh;
%             C = ada_test_classify_set(CASCADE, PREMAT);
% 
%             [tpr fpr FPs TP] = rocstats(C, gt, 'TPR', 'FPR', 'FPlist', 'TP');
%             %disp([num2str(learners) ' learners, Di=' num2str(tpr) ', Fi=' num2str(fpr) ', #FP = ' num2str(length(FPs)) '.  CASCADE stage ' num2str(i) ' learner ' num2str(ti) ' applied to TEST set.'  ]);               
% 
%             ROCDATA(count,:) = [tpr fpr length(FPs) TP];
%             count = count + 1;
%         end
%         CASCADE(s).thresh = 0;
%         if s > 1
%             CASCADE = CASCADE(1:s-1);
%         end
%    end
% end

function [varargout] = rocstats(test, gt, varargin)
%ROCSTATS Provides ROC detection statistical analysis.
%
%   [S1, S2, ..., SN] = rocstats(test, gt, 'S1', 'S2', ..., 'SN') returns
%   classification and receiver-operating characteristics (ROC) for a
%   two-class classification problem.  Given a hypothesis vector 'test' and
%   a ground truth vector 'gt' (vectors elements contain 0 or 1 for
%   negative and positive examples, resp.), the following statistics are
%   provided (see http://en.wikipedia.org/wiki/Receiver_operating_characteristic 
%   for details on these statistics):
%
%  'TP'  true positive count (hits)
%  'TN'  true negative count (correct rejections)
%  'FP'  false positive count (false alarms, type I errors)
%  'FN'  false negative count (type II errors)
%  'TPR' true positive rate (hit rate, recall, sensitivity)
%  'TNR' true negative rate
%  'FPR' false  positive rate (false alarm rate, fall-out)
%  'ACC' accuracy
%  'SPC' specificity
%  'PPV' positive predictive value (precision)
%  'NPV' negative predictive value
%  'FDR' false discovery rate
%  'MCC' Matthews Correlation Coefficient
%
%   The user may also request a list of the indices of types of errors made:
%   'TPlist'    list of true positives
%   'TNlist'    list of true negatives
%   'FPlist'    list of false positives
%   'FNlist'    list of false negatives
%
%   example:  [tn tpr acc] = rocstats([1 1 0],[0 1 0], 'TN', 'TPR', 'ACC')
%   returns:  tn = 1  tpr = 1  acc = .667
%
%   Copyright 2008 Kevin Smith
%
%   See also STRUCT

% convert any matices or tensors to vectors
test = test(:); gt = gt(:);


for k = 1:nargin-2
   
    switch varargin{k}
        case 'TP'
            % true positive count (hits)
            TP = sum(test & gt);
            varargout{k} = TP; 
        case 'TN'
            % true negative count (correct rejections)
            TN = sum(~test & ~gt);
            varargout{k} = TN;
        case 'FP'
            % false positive count (false alarms, type I errors)
            FP = sum(test & ~gt);
            varargout{k} = FP;
        case 'FN'
            % false negative count (type II errors)
            FN = sum(~test & gt);
            varargout{k} = FN;
        case 'TPR'
            % true positive rate (hit rate, recall, sensitivity)
            if ~exist('TP','var'); TP = sum(test & gt); end
            TPR = TP / sum(gt);
            varargout{k} = TPR;   
        case 'TNR'
            % true negative rate
            if ~exist('TN','var'); TN = sum(~test & ~gt); end
            TNR = TN / sum(~gt);
            varargout{k} = TNR;  
        case 'FPR'
            % false  positive rate (false alarm rate, fall-out)
            if ~exist('FP', 'var'); FP = sum(test & ~gt); end
            FNR = FP/ sum(~gt);
            varargout{k} = FNR;
        case 'ACC'
            % accuracy
            if ~exist('TP','var'); TP = sum(test & gt); end
            if ~exist('TN','var'); TN = sum(~test & ~gt); end
            ACC = (TP + TN) / length(gt);
            varargout{k} = ACC;
        case 'SPC'
            % specificity
            if ~exist('TN','var'); TN = sum(~test & ~gt); end
            if ~exist('FP','var'); FP = sum(test & ~gt); end
            SPC = TN / (FP + TN);
            varargout{k} = SPC;
        case 'PPV'
            % positive predictive value (precision)
            if ~exist('TP','var'); TP = sum(test & gt); end
            if ~exist('FP','var'); FP = sum(test & ~gt); end
            PPV = TP / (TP + FP);
            varargout{k} = PPV;
        case 'NPV'
            % negative predictive value
            if ~exist('TN','var'); TN = sum(~test & ~gt); end
            if ~exist('FN','var'); FN = sum(~test & gt); end
            NPV = TN / (TN + FN);
            varargout{k} = NPV;
        case 'FDR'
            % false discovery rate
            if ~exist('TP','var'); TP = sum(test & gt); end
            if ~exist('FP','var'); FP = sum(test & ~gt); end
            FDR = FP / (FP + TP);
            varargout{k} = FDR;
        case 'MCC'
            % Matthews Correlation Coefficient
            if ~exist('TP','var'); TP = sum(test & gt); end
            if ~exist('TN','var'); TN = sum(~test & ~gt); end
            if ~exist('FP','var'); FP = sum(test & ~gt); end
            if ~exist('FN','var'); FN = sum(~test & gt); end
            MCC = (TP*TN - FP*FN) / sqrt( sum(test)*sum(~test)*sum(gt)*sum(~gt));
            varargout{k} = MCC;
        case 'TPlist'
            TPlist = find(test & gt);
            varargout{k} = TPlist;
        case 'TNlist'
            TNlist = find(~test & ~gt);
            varargout{k} = TNlist;
        case 'FPlist'
            FPlist = find(test & ~gt);
            varargout{k} = FPlist;
        case 'FNlist'
            FNlist = find(~test & gt);
            varargout{k} = FNlist;
    end

end
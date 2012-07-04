function priors = assignPriors(D, Dlist, trkSeq, SL, TMAX)  %#ok<INUSL>

% priors = cell(size(D));
% for t = 1:TMAX
%     if t == 1
%         priorList = ones(1,numel(Dlist{t})) ./ numel(Dlist{t});
%     else
%         priorList = zeros(1,numel(Dlist{t})) ./ numel(Dlist{t});
%         for i = 1:length(Dlist{t})
%             trkID = D(Dlist{t}(i)).ID;
%             if trkID ~= 0
%                 ind = find(trkSeq{trkID} == Dlist{t}(i));
% 
%                 if ind ~= 1
%                     prevID =  trkSeq{trkID}( ind - 1  );
%                     priorList(i) = sum(sum(SL{t-1} == prevID));
%                 end
%             end
%         end
% 
%         minval = min( priorList( find (priorList))); %#ok<FNDSB>
%         priorList(find(priorList == 0)) = minval; %#ok<FNDSB>
%         % set zeros in the prior list to be the min value
% 
%     end
%     priorList = priorList / sum(priorList);
%     priors{t} = priorList;
% end

priors = cell(size(D));
for t = 1:TMAX
    priorList = ones(1,numel(Dlist{t})) ./ numel(Dlist{t});
    priors{t} = priorList;
end

function C = ada_classify_cascade(CASCADE, II, offset)
%
%
%
%
%
%

if nargin == 2
    offset = [0 0 ];
end

C = 0;


for i = 1:length(CASCADE)    
    if ~ada_classify_strong(CASCADE(i).CLASSIFIER, II, offset, CASCADE(i).threshold)
        %disp(['rejected at stage ' num2str(i)]);
        return
    end    
end

C = 1;


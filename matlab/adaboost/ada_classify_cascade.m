function C = ada_classify_cascade(CASCADE, DATA, offset)
%C = ada_classify_cascade(CASCADE, DATA, offset)
%
%  DATA contains integral images, spedges, etc...
%
%
%

if nargin == 2
    offset = [0 0 ];
end

C = 0;


for i = 1:length(CASCADE)    
    if ~ada_classify_strong(CASCADE(i).CLASSIFIER, DATA, offset, CASCADE(i).threshold)
        %disp(['rejected at stage ' num2str(i)]);
        return
    end    
end

C = 1;


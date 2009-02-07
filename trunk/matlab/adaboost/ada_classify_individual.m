function C = ada_classify_individual(CLASSIFIER, I, LEARNERS)
%function C = ada_classify_individual(CLASSIFIER, I, LEARNERS, TRAIN,i)
%   TRAIN, and i are (i think temporary) and not necessary
%
%   C = ada_classify_individual(CLASSIFIER, I, LEARNERS)
%
%   seems to work for haar, haven't checked yet for spedges
%
%


% if we are passed a single stage instead of a cascade, make it appear as a
% similar structure to a cascade
if strcmp(CLASSIFIER.type, 'SINGLE');
    CASCADE.CLASSIFIER = CLASSIFIER;
else
    CASCADE = CLASSIFIER;
end

% compute the integral image and spedge features only once
for l = 1:length(LEARNERS)
    if strcmp(LEARNERS(l).feature_type, 'haar') 
        II = integral_image(I);
        II = II(:);
    end

    if strcmp(LEARNERS(l).feature_type, 'spedge')
        sp = spedges(I, LEARNERS(l).angles, LEARNERS(l).sigma);
        %SP = sp.spedges(:);
    end
    
end
   


% initialize C to zero
C = 1;

    
for s = 1:length(CASCADE)
    
    
    for l = 1:length(CASCADE(s).CLASSIFIER.weak_learners)
        
        switch CASCADE(s).CLASSIFIER.weak_learners{l}.type
            
            case 'haar'
                hinds = CASCADE(s).CLASSIFIER.weak_learners{l}.hinds;
                hvals = CASCADE(s).CLASSIFIER.weak_learners{l}.hvals;
                f(l) = ada_haar_response(hinds, hvals, II);
                
%                 b(l) = TRAIN.responses.get(i, CASCADE(s).CLASSIFIER.weak_learners{l}.index);
%                 
%                 if abs(f(l) - b(l)) > 1e-03
%                     keyboard;
%                 end
                
                polarity(l) = CASCADE(s).CLASSIFIER.polarity(l);
                theta(l) = CASCADE(s).CLASSIFIER.theta(l);
                
            case 'spedge'
                %f = SP(CASCADE(s).CLASSIFIER.weak_learners{l}.index);
                
                f = sp(CASCADE(s).CLASSIFIER.weak_learners{l}.row, CASCADE(s).CLASSIFIER.weak_learners{l}.col,CASCADE(s).CLASSIFIER.weak_learners{l}.sigma, CASCADE(s).CLASSIFIER.weak_learners{l}.angle);                
                
                polarity(l) = CASCADE(s).CLASSIFIER.polarity(l);
                theta(l) = CASCADE(s).CLASSIFIER.theta(l);
        end
            
        
        % weak classification
        h(l) = polarity(l) * f(l) < polarity(l) * theta(l);
        
        
%         b(l) = polarity(l) * f(l) < polarity(l) * theta(l);
    end
    
%     if ~isequal(h, b)
%         disp('classifications based on computed features (f) and from bigmatrix (b) do not agree.');
%         keyboard;
%     else
%         disp('classifications based on computed features (f) and from bigmatrix (b) seems to agree.');
%         disp(['h = ' num2str(h)]);
%         disp(['b = ' num2str(b)]);
%     end
    
%     disp(['polarity ' num2str(polarity)]);
%     disp(['f        ' num2str(f)]);
%     disp(['theta    ' num2str(theta)]);
%     disp(['h        ' num2str(h)]);

    
    % compute the strong classification
    C = sum(h .* CASCADE(s).CLASSIFIER.alpha)  >  ( .5 * sum(CASCADE(s).CLASSIFIER.alpha) * CASCADE(s).threshold) ;
        
    % if the current stage generates a rejection, we can stop
    if C == 0
        break;
    end
end
    
    


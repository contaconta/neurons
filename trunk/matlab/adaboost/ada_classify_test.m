function C = ada_classify_test(CLASSIFIER, ind, LIST, LEARNERS)
%
%
%   C = ada_classify_test(CLASSIFIER, SET, LEARNERS)
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

% % compute the integral image and spedge features only once
% for l = 1:length(LEARNERS)
%     if strcmp(LEARNERS(l).feature_type, 'haar') 
%         II = integral_image(I);
%         II = II(:);
%     end
% 
%     if strcmp(LEARNERS(l).feature_type, 'spedge')
%         sp = spedges(I, LEARNERS(l).angles, LEARNERS(l).sigma);
%         SP = sp.spedges(:);
%     end
%     
% end
   
% initialize C to zero
C = 1;

    
for s = 1:length(CASCADE)
    
    %h = zeros(size(CASCADE(s).feature_index));
    
    h_ind = 1;   s_ind = 1;     % indexes of the haar and spedge features of this stage
    
    % compute the necessary features for this stage and use them to do weak
    % classification on the image (store in h)  
    for l = 1:length(CASCADE(s).CLASSIFIER.learner_type)
        learner_type = CASCADE(s).CLASSIFIER.learner_type{l};

        switch learner_type
            case 'haar'
                
                hinds = CASCADE(s).CLASSIFIER.haar(h_ind).hinds;
                hvals = CASCADE(s).CLASSIFIER.haar(h_ind).hvals;
                %f(l) = ada_haar_response(hinds, hvals, II);
                %f(l) = SET.responses.get(ind, CASCADE(s).CLASSIFIER.feature_index(l));
                f(l) = LIST(ind, l);
                
                
                polarity(l) = CASCADE(s).CLASSIFIER.haar(h_ind).polarity;
                theta(l) = CASCADE(s).CLASSIFIER.haar(h_ind).theta;
                
                h_ind = h_ind + 1;

            case 'spedge'

                %f = SP(CASCADE(s).CLASSIFIER.spedge(s_ind).index);
                %f(l) = SET.responses.get(ind, CASCADE(s).CLASSIFIER.feature_index(l));
                f(l) = LIST(ind, l);
                
                polarity(l) = CASCADE(s).CLASSIFIER.spedge(s_ind).polarity;
                theta(l) = CASCADE(s).CLASSIFIER.spedge(s_ind).theta;
                s_ind = s_ind + 1;
        end
        
        % weak classification
        h(l) = polarity(l) * f(l) < polarity(l) * theta(l);
    end
    
    % compute the strong classification
    C = sum(h .* CASCADE(s).CLASSIFIER.alpha)  >  ( .5 * sum(CASCADE(s).CLASSIFIER.alpha) * CASCADE(s).threshold) ;
        
    % if the current stage generates a rejection, we can stop
    if C == 0
        break;
    end
end
    
    
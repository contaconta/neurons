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
if strcmp(CLASSIFIER(1).type, 'SINGLE');
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
        edgelist.sigma = 0;
        edgelist.EDGE = I;
    end
end
   
% initialize C to zero
C = 1;

    
for s = 1:length(CASCADE)
    for l = 1:length(CASCADE(s).CLASSIFIER.weak_learners)
        switch CASCADE(s).CLASSIFIER.weak_learners{l}.type
            
            case 'intmean'
                f(l) = ada_intmean_response(I);
                polarity(l) = CASCADE(s).CLASSIFIER.polarity(l);
                theta(l) = CASCADE(s).CLASSIFIER.theta(l);
                
            case 'intvar'
                f(l) = ada_intvar_response(I);
                polarity(l) = CASCADE(s).CLASSIFIER.polarity(l);
                theta(l) = CASCADE(s).CLASSIFIER.theta(l);
                
            case 'haar'
                hinds = CASCADE(s).CLASSIFIER.weak_learners{l}.hinds;
                hvals = CASCADE(s).CLASSIFIER.weak_learners{l}.hvals;
                f(l) = ada_haar_response(hinds, hvals, II);

                polarity(l) = CASCADE(s).CLASSIFIER.polarity(l);
                theta(l) = CASCADE(s).CLASSIFIER.theta(l);
                
            case 'spedge'
                angle = CASCADE(s).CLASSIFIER.weak_learners{l}.angle;
                sigma = CASCADE(s).CLASSIFIER.weak_learners{l}.sigma;
                row = CASCADE(s).CLASSIFIER.weak_learners{l}.row;
                col = CASCADE(s).CLASSIFIER.weak_learners{l}.col;
                
                % if we've already computed EDGE for SIGMA, use it, otherwise use I
                ind = find([edgelist(:).sigma] == sigma,1);
                if ~isempty(ind)
                    EDGE = edgelist(ind).EDGE;
                    [f(l) EDGE] = single_spedge(angle, sigma, row, col, EDGE, 'edge');
                    %disp('using a previously stored edge');
                else
                    [f(l) EDGE] = single_spedge(angle, sigma, row, col, I);
                    %disp('using a newly computed edge');
                end
                
                % store EDGE for this SIGMA value
                if isempty(find([edgelist(:).sigma] == sigma,1));
                    edgelist(length(edgelist)+1).sigma = sigma;
                    edgelist(length(edgelist)).EDGE = EDGE;
                end

                polarity(l) = CASCADE(s).CLASSIFIER.polarity(l);
                theta(l) = CASCADE(s).CLASSIFIER.theta(l);
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
    


%         sp = spedges(I, LEARNERS(l).angles, LEARNERS(l).sigma);

%                 for a = 1:length(LEARNERS); if strcmp(LEARNERS(a).feature_type, CASCADE(s).CLASSIFIER.weak_learners{l}.type); L_ind = a; end; end;
%                 angle_ind = find(LEARNERS(L_ind).angles ==CASCADE(s).CLASSIFIER.weak_learners{l}.angle);
%                 sigma_ind = find(LEARNERS(L_ind).sigma ==CASCADE(s).CLASSIFIER.weak_learners{l}.sigma);
%                 f(l) = sp.spedges(angle_ind, sigma_ind, CASCADE(s).CLASSIFIER.weak_learners{l}.row, CASCADE(s).CLASSIFIER.weak_learners{l}.col);                
                


%                 b(l) = TRAIN.responses.get(i, CASCADE(s).CLASSIFIER.weak_learners{l}.index);
%                 
%                 if abs(f(l) - b(l)) > 1e-03
%                     keyboard;
%                 end
        
        
%         b(l) = polarity(l) * f(l) < polarity(l) * theta(l);



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

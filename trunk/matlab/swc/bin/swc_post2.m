function Plist = swc_post2(W, LABELS, GT, pixelList, Iraw, R, varargin)
% function Plist = swc_posterior(W, LABELS, model, minI, maxI, rescaleData, IGroundTruth, ...
%                       useGroundTruth, pixelList, Iraw, R, varargin)                  
%
% Plist = swc_post(W, LABELS, model, minI, maxI, B, pixelList, Iraw, R, Plist)
% Plist = swc_post(W, LABELS, model, minI, maxI, B, pixelList, Iraw, R, 'init')



%if strcmp(varargin{1},'init')
if ischar(varargin{1})
    % initialize Plist
    Plist = zeros(1, size(W,1));
    NODES = 1:size(W,1);
else
    Plist = varargin{1};
    NODES = R(:)';
end


global minI maxI model G0; %#ok<TLEV>

% traverse either the entire graph, or nodes in R that have changed
for v = NODES

    
            % svm
            nodes = graphtraverse(G0,v, 'depth', 1);
            pixels = Iraw(cell2mat(pixelList(nodes)'));
            [predicted_label, accuracy, pb] = getLikelihood(pixels,model,minI,maxI,true);
            %Plist(v) = pb(LABELS(v));
            % pb = [bg boundary mitochondria]
            Plist(v) = pb(LABELS(v));
    
            
    
   % if strcmp(modelCell{1}, 'perfect')          
   %     GT = modelCell{2};
        %Plist(v) = -5 + 6*(GT(v)==LABELS(v));    % Plist(v) = -5 if not match, = 1 if match 
% <<<<< BELOW IS GOOD FOR THE PERFECT GROUND TRUTH >>>>>
%        Plist(v) = -1 + 1.01*(GT(v)==LABELS(v));    % Plist(v) = -1 if not match, = .01 if match 
   % end

%     if stecmp(modelCell{1}, 'svm')
%         nodes = graphtraverse(W,v, 'depth', 1);
%         pixels = Iraw(cell2mat(pixelList(nodes)'));
%         [predicted_label, accuracy, pb] = getLikelihood(pixels, ...
%                                                         model,minI,maxI,rescaleData);
%         Plist(v) = pb(LABELS(v));
%     end
end

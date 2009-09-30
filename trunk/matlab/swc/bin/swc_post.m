function P = swc_post(W, LABELS, model, minI, maxI, pixelList, Iraw, V0, varargin)
%
% P = swc_post(W, LABELS, model, minI, maxI, B, pixelList, Iraw, V0, P)
% P = swc_post(W, LABELS, model, minI, maxI, B, pixelList, Iraw, V0, 'init')


% edges = find(triu(G0) == 1)';
% [vi, vj] = ind2sub(size(G0), edges);

if model == 0
    useGroundTruth=true;
else
    useGroundTruth=false;
end
IGroundTruth = imread('images/annotation0002.png');
IGroundTruth = IGroundTruth(:,:,1);
IGroundTruth = IGroundTruth(1:480,1:640);

% TODO AL : Debug this function

 
if strcmp(varargin{1},'init')
    % initialize
    P = zeros(1, size(W,1));

    for v = 1:size(W,1)

        nodes = graphtraverse(W,v, 'depth', 1);

        if useGroundTruth
            lpixels = cell2mat(pixelList(nodes)');
            P(v) = (getMostFrequentLabel(lpixels,IGroundTruth) == LABELS(v));
        else
            pixels = Iraw(cell2mat(pixelList(nodes)'));
            [predicted_label, accuracy, pb] = getLikelihood(pixels, model,minI,maxI);
            P(v) = pb(LABELS(v));
        end
    end

else
    P = varargin{1};
    
    for v = V0(:)

    	nodes = graphtraverse(W,v, 'depth', 1);

        if useGroundTruth
            lpixels = cell2mat(pixelList(nodes)');
            P(v) = (getMostFrequentLabel(lpixels,IGroundTruth) == LABELS(v));
        else
            pixels = Iraw(cell2mat(pixelList(nodes)'));
            [predicted_label, accuracy, pb] = getLikelihood(pixels, model,minI,maxI);
            P(v) = pb(LABELS(v));
        end
    end
    
end

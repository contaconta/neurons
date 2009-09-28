function P = swc_post(W, LABELS, model, minI, maxI, pixelList, Iraw, V0, varargin)
%
% P = swc_post(W, LABELS, model, minI, maxI, B, pixelList, Iraw, V0, P)
% P = swc_post(W, LABELS, model, minI, maxI, B, pixelList, Iraw, V0, 'init')


% edges = find(triu(G0) == 1)';
% [vi, vj] = ind2sub(size(G0), edges);

useGroundTruth=true;
IGroundTruth = imread('/home/alboot/usr/work/EM/raw_mitochondria/annotation/annotation0002.png');

% TODO AL : Debug this function
keyboard;
 
if strcmp(varargin{1},'init')
    % initialize
    P = zeros(1, size(W,1));

    for v = 1:size(W,1)

        nodes = graphtraverse(W,v, 'depth', 1);

        if useGroundTruth
            P(v) = getMostFrequentLabel(pixelList(nodes)',IGroundTruth);
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

        pixels = Iraw(cell2mat(pixelList(nodes)'));
        [predicted_label, accuracy, pb] = getLikelihood(pixels, model,minI,maxI);
        P(v) = pb(LABELS(v));
    end
    
end

function l = getMostFrequentLabel(pixelList, IGroundTruth)

count = 0;
for p in pixelList
if(IGroundTruth(p))
    count = count + 1;
else
    count = count - 1;
end
end

l = count > 0;

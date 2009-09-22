function P = swc_posterior(W, LABELS, model, minI, maxI, B, pixelList, Iraw)


% edges = find(triu(G0) == 1)';
% [vi, vj] = ind2sub(size(G0), edges);


P = 0;

for v = 1:size(W,1)

    nodes = graphtraverse(W,v, 'depth', 1);
    
    pixels = Iraw(cell2mat(pixelList(nodes)'));

    [predicted_label, accuracy, pb] = getLikelihood(pixels, model,minI,maxI);
    
    pbv = pb(LABELS(v));
    
    P = P + pbv * B;
end


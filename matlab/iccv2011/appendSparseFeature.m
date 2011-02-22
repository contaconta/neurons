function f = appendSparseFeature(fid, x,y,w,s,xc,yc,rotFlag,transFlag, W, H)

if ~exist('W', 'var')
    W = -1;
end
if ~exist('H', 'var')
    H = -1;
end

numG = numel(x);

f = [num2str(numG) sprintf(' %f %f', xc, yc)];

for k = 1:numG       
    f = [f sprintf(' %f %f %d %d %d ',  w(k), s(k), x(k), y(k) )]; %#ok<*AGROW>
    
    
    
end

f = [f sprintf('%d %d %d %d', rotFlag, transFlag, W, H)];

fprintf(fid, [f '\n']);

%keyboard;
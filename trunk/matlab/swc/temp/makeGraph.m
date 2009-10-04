function makeGraph(labelFilenm, outFilenm,closeMatlab)
% Create binary files containing a list of neighbors for each superpixel in the specified image

if nargin<3 || isempty(closeMatlab)
 closeMatlab = 0;
end


addpath('../utils');
addpath('../bin');

% Read files containing the labels
L = readRKLabel(labelFilenm, [1536 2048])';

[G0, G0list] = adjacency(L);

% Output file
fid = fopen(outFilenm, 'w');
for j=1:length(G0list)
  fprintf(fid, '%d ', j);
  for k=1:length(G0list{j})
    fprintf(fid, '%d ', G0list{j}(k));
  end
  fprintf(fid, '\n');
end
fclose(fid);

if closeMatlab
  quit
end
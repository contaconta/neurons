function writeLIBSVMfeaturevector(featureVector, L, superpixels, mito, filenm, path, varargin)

if nargin > 6
    N = varargin{1};
else
    N = 200;  % number of feature vectors to extract for this image
    %N = 20;
end

%% seed the random # generator
st = RandStream.create('mt19937ar','seed',5489);  RandStream.setDefaultStream(st);  %rand('twister', 100);    % seed for Matlab 7.8 (?)


fid = fopen([path filenm], 'a');


m = find(mito == 1);  % contains indexes of mitochondria-labeled superpixels
n = find(mito == 0);  % contains indexes of non-mitochondria-labeled superpixels



mlist = randsample(m, N);
%mlist = m;                     % uncomment to list EVERY superpixel

for s = mlist
    
    fprintf(fid, '%d ', 2);
    
    for i = 1:size(featureVector,2)-1
        fprintf(fid, '%d:%g ', i, featureVector(s,i));
    end
    
    i = size(featureVector,2);
    fprintf(fid, '%d:%g\n', i, featureVector(s,i));
    
%     for i = 1:24
%         fprintf(fid, '%d:%d ', i, featureVector(s,i));
%     end
%     
%     for i = 25:299
%         fprintf(fid, '%d:%f ', i, featureVector(s,i));
%     end
%     i = 300;
%     fprintf(fid, '%d:%f\n', i, featureVector(s,i));
end


nlist = randsample(n, N);
%nlist = n;                     % uncomment to list EVERY superpixel

for s = nlist
    
    fprintf(fid, '%d ', 0);
    
    for i = 1:size(featureVector,2)-1
        fprintf(fid, '%d:%g ', i, featureVector(s,i));
    end
    
    i = size(featureVector,2);
    fprintf(fid, '%d:%g\n', i, featureVector(s,i));
    
%     for i = 1:24
%         fprintf(fid, '%d:%d ', i, featureVector(s,i));
%     end
%     
%     for i = 25:299
%         fprintf(fid, '%d:%g ', i, featureVector(s,i));
%     end
%     
%     i = 300;
%     fprintf(fid, '%d:%f\n', i, featureVector(s,i));
end


fclose(fid);
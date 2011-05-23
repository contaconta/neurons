function  writeMultiPageTiff(V, fname)
%writeMultiPageTiff(V, fname)
%
%
%

% if the file already exists, delete it so we do not append to it!
if exist(fname, 'file')
    delete(fname);
end

for i = 1:size(V,3)
    imwrite(V(:,:,i), fname ,'tif', 'Compression', 'none', 'WriteMode', 'append');
end
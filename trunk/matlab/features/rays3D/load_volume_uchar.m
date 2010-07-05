function V = load_volume_uchar(filename, SIZE)


fid = fopen(filename,'r');
%V = fread(fid,[SIZE(2) SIZE(1) SIZE(3)],'uchar');
V = fread(fid,prod(SIZE),'uchar');

V = reshape(V, SIZE);

for i = 1:size(V,3)
    V(:,:,i) = V(:,:,i)';
end

fclose(fid);
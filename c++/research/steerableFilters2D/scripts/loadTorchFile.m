function data = loadTorchFile(filename)
%% loads data sotored in Torch format
f = fopen(filename,'r');
r = fscanf(f, '%f',1);
c = fscanf(f, '%f',1);
data = fscanf(f, '%f');
fclose(f);
data = reshape(data, c, r)';


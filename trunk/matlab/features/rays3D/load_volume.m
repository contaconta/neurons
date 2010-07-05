function V = load_volume(folder)

d = dir([folder '/*.png']);

if ~ isempty(d)
    filename = [folder '/' d(1).name];
    I = imread(filename);

    V = zeros([size(I) length(d)]);
    V(:,:,1) = I;
    
    for i = 2:length(d)
        
        filename = [folder '/' d(i).name];
        I = imread(filename);
        V(:,:,i) = I;
        


    end
    
else
    
end


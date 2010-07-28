function data_cleaning()


d = dir([pwd '/' '*.tif']);



for i = 1:length(d)
    
    fname = d(i).name;
    disp(['cleaning ' fname]);
    
    % read the tiff, clean it
    V = readMultiPageTiff(fname);
    
    % delete the current copy
    delete(fname);
    
    % write the cleaned copy
    writeMultiPageTiff(V, fname);
    
end



function writeLIBSVMclassificationVector(featureVector, superpixels, filenm, path, mito)



fid = fopen([path filenm], 'a');

for s = superpixels
    
    if mito(s) == 1
        
         fprintf(fid, '%d ', 2);

        for i = 1:size(featureVector,2)-1
            fprintf(fid, '%d:%g ', i, featureVector(s,i));
        end

        i = size(featureVector,2);
        fprintf(fid, '%d:%g\n', i, featureVector(s,i));
    
    else
    
        fprintf(fid, '%d ', 0);

        for i = 1:size(featureVector,2)-1
            fprintf(fid, '%d:%g ', i, featureVector(s,i));
        end

        i = size(featureVector,2);
        fprintf(fid, '%d:%g\n', i, featureVector(s,i));

    end
end


fclose(fid);
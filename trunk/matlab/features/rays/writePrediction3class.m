function writePrediction3class(path, filenm, predictions, L, label_types)

[label_types, inds] = sort(label_types);

fid = fopen([path filenm], 'w');

fprintf(fid, 'labels %d %d %d\n', label_types(1), label_types(2), label_types(3));


for s = 1:size(predictions,1)    
    fprintf(fid, '%d %g %g %g\n', L(s), predictions(s, inds(1)), predictions(s, inds(2)), predictions(s, inds(3)));
end


fclose(fid);
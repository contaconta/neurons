function writePairwisePrediction(path, filenm, r, c, predictions, L, label_types)

[label_types, inds] = sort(label_types);

fid = fopen([path filenm], 'w');

fprintf(fid, 'labels %d %d\n', label_types(1), label_types(2));


for s = 1:size(predictions,1) 
    fprintf(fid, '%d %d %d %g %g\n', r(s), c(s), L(s), predictions(s, inds(1)), predictions(s, inds(2)));
end


fclose(fid);
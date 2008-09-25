function Mavg = vj_mean_mask(CASCADE)

c = 1; 
for j = 1:length(CASCADE); 
    for i=1:size(CASCADE(j).CLASSIFIER.feature_descriptor,1); 
        M(:,:,c) = CASCADE(j).CLASSIFIER.alpha(i)*vj_plot_haar_feature(CASCADE(j).CLASSIFIER.feature_descriptor(i,:), CASCADE(j).CLASSIFIER.polarity(i), [24 24], 'D', 0); 
        c = c+1; 
    end; 
end;

Mavg = mean(M,3);
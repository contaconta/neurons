p_settings;

I = imread('test_mito.png');
II = integral_image(I);


% define the weak learners
LEARNERS = p_EnumerateLearners(LEARNERS, DATASETS.IMSIZE);
f_mat = zeros(1, length(LEARNERS.list));
f_mex = f_mat;

for l = 1:length(LEARNERS.list)
    
    visualize_haar_feature(LEARNERS.list{l}, size(I), I);
    
    % mex haar calculation
    f_mex(l) = mexRectangleFeature({II}, LEARNERS.list(l));
      
    
    % matlab direct calculartion
    f_mat(l) = compute_haar_feature(I, LEARNERS.list{l});
    
    if f_mex(l) ~= f_mat(l)
        disp([' feature responses do not agree for feature ' num2str(l) '!']);
        disp([' f_mex: ' num2str(f_mex(l)) '   f_mat: ' num2str(f_mat(l)) ]);
        keyboard;
    end
end

disp('feature responses agree!');
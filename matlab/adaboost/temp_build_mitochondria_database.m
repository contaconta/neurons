pos_train_path = '/osshare/Work/Data/mitochondria24/train/pos/';
neg_train_path = '/osshare/Work/Data/mitochondria24/train/neg/';
pos_test_path  = '/osshare/Work/Data/mitochondria24/test/pos/';
neg_test_path  = '/osshare/Work/Data/mitochondria24/test/neg/';


IMSIZE = [24 24];           % THE SIZE EACH TRAINING EXAMPLE WILL BE RESIZED TO!!!!

rand('twister', 100);       % seed the random variable

image_path = '/osshare/Work/Data/raw_mitochondria/originals/';
annotation_path = '/osshare/Work/Data/raw_mitochondria/annotation/';


ECC_THRESH = 5;

d_images = dir([image_path '*.png']);
d_annotations = dir([annotation_path '*.png']);

NEGATIVE_EXAMPLES = 5000;

BORDER = .2;

count = 1;  noncount = 1;

% loop through images
for im = 1:length(d_images)
    
    imfilename = [image_path d_images(im).name];
    gtfilename = [annotation_path d_annotations(im).name];
    
    I = imread(imfilename);  disp(['read ' imfilename]);
    GT = imread(gtfilename); disp(['read ' gtfilename]);
    
    ISIZE = size(I);
    
    I = rgb2gray(mat2gray(I));
    GT = rgb2gray(mat2gray(GT));
    GT = GT > .5;
    
    
    L = bwlabel(GT);
    stats = regionprops(L, 'Orientation', 'BoundingBox', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
    
    
    num_mito = max(max(L));
    

    %======================================================================
    %  POSITIVE EXAMPLES
    %======================================================================
    
    % loop through the mitochondria, and extract them!
    for i = 1:num_mito
        
        BB = stats(i).BoundingBox;
        
        if (BB(1) == 1) || (BB(2) == 1) || (BB(2) + BB(3) >= ISIZE(2)) || (BB(1)+BB(4) >= ISIZE(1))
            continue;
        end
        
        
        Added = BB(3:4) * (BORDER);
        BB(3:4) = BB(3:4) * (1 + BORDER);
        
        BB(1:2) = BB(1:2) - round(Added/2);
        
        
        if BB(3) >= BB(4)
            ADD = round( abs(BB(3) - BB(4))/2);
            BB(2) = max(0, BB(2) - ADD);
            BB(4) = BB(3);
           
        else
            ADD = round( abs(BB(4) - BB(3))/2);
            BB(1) = max(0, BB(1) - ADD);
            BB(3) = BB(4);
        end
        
        
        E = imcrop(I, BB);
        W = stats(i).BoundingBox(3);
        H = stats(i).BoundingBox(4);

        if stats(i).MajorAxisLength/stats(i).MinorAxisLength > ECC_THRESH    %max([W H])/min([W H]) > 2.5
            disp('this one is too elongated!');
            disp(['TOO ELONGATED! Maj Axis: ' num2str(stats(i).MajorAxisLength) 'Minor Axis: ' num2str(stats(i).MinorAxisLength) '     Ratio: ' num2str(stats(i).MajorAxisLength/stats(i).MinorAxisLength)]);
        else
            %disp(['boundingbox: ' num2str(stats(i).BoundingBox) '  Eccentricity: ' num2str(stats(i).Eccentricity) ]);
            disp(['Maj Axis: ' num2str(stats(i).MajorAxisLength) 'Minor Axis: ' num2str(stats(i).MinorAxisLength) '     Ratio: ' num2str(stats(i).MajorAxisLength/stats(i).MinorAxisLength)]);
        
            E = imresize(E, IMSIZE);
            imshow(E);       set(gca, 'Position', [0 0 1 1]);     
            pause(0.1);
        
            nstr = number_into_string(count, 10000);

            test_file1 = [pos_test_path 'mito' nstr '.png'];
            imwrite(E, test_file1, 'PNG');
            disp(['wrote  ' test_file1 ]); 

            E2 = imrotate(E, 180);
            train_file2 = [pos_train_path 'mito' nstr '.png'];
            imwrite(E2, train_file2, 'PNG');
            disp(['wrote  ' train_file2 ]); 
            
            count = count + 1;
            
            nstr = number_into_string(count, 10000);
            E2 = imrotate(E, 90);
            test_file1 = [pos_test_path 'mito' nstr '.png'];
            imwrite(E, test_file1, 'PNG');
            disp(['wrote  ' test_file1 ]); 

            E2 = imrotate(E, 270);
            train_file2 = [pos_train_path 'mito' nstr '.png'];
            imwrite(E2, train_file2, 'PNG');
            disp(['wrote  ' train_file2 ]); 
            
            count = count + 1;
        end
        

    end
    
    
%     %======================================================================
%     %  NEGATIVE EXAMPLES
%     %======================================================================
%     
%     N_NEEDED = ceil(NEGATIVE_EXAMPLES / length(d_images));
%     
%     Isize = size(I);
%     
%     W_MU = 65 * (1 + BORDER);
%     H_MU = 65 * (1 + BORDER);
%     W_STD = 30;  W_VAR = W_STD^2;
%     H_STD = 30;  H_VAR = H_STD^2;
% 
%     
%     while noncount < im*N_NEEDED
%     
%         % select a random location
%         W = round(gsamp(W_MU, W_VAR, 1));
%         if W < 20; W = 20;   end
%         H = W;   %round(gsamp(H_MU, H_VAR, 1));
%         X = ceil(  (Isize(2) - W)  *rand(1));
%         Y = ceil(  (Isize(1) - H) * rand(1));
% 
%         BB = [X Y W H];
%         
%         % check to make sure we don't overlap an annotation
%         
%         E = imcrop(I,BB);
%         GTCROP = imcrop(GT, BB);
%         
%                 
%         if (sum(sum(GTCROP)) / numel(GTCROP)) > .2
%             disp('...picked a spot with mitochondria!');
%         else
%             E = imresize(E, IMSIZE);
%             %GTCROP = imresize(GTCROP, IMSIZE);
% 
% %             imshow(E);          
% %             pause(0.1);
%             
%             nstr = number_into_string(noncount, 10000);
% 
%             test_file1 = [neg_test_path 'nonmito' nstr '.png'];
%             imwrite(E, test_file1, 'PNG');
%             disp(['wrote  ' test_file1 ]); 
% 
%             E2 = imrotate(E, 180);
%             train_file2 = [neg_train_path 'nonmito' nstr '.png'];
%             imwrite(E2, train_file2, 'PNG');
%             disp(['wrote  ' train_file2 ]); 
%             noncount = noncount + 1;
%         end
%     end
%         
%         
%     
%     % store the ground truth
%     %I1 = imoverlay(I, GT);
%     %imwrite(I1, ['/osshare/Work/Data/raw_mitochondria/GT' number_into_string(im,100) '.png'] );
%     
end



%disp('...loading GT.mat');  
%load /osshare/Work/matlab_neuron_project/cvpr_tracking/GT.mat;

pos_train_path = '/osshare/Work/Data/nucleus_training_rotated/train/pos/';
neg_train_path = '/osshare/Work/Data/nucleus_training_rotated/train/neg/';
pos_test_path  = '/osshare/Work/Data/nucleus_training_rotated/test/pos/';
neg_test_path  = '/osshare/Work/Data/nucleus_training_rotated/test/neg/';



%BORDER = 8;  %8;
IMSIZE = [24 24];
ex_num = 1;

%==========================================================================
%  CREATE POSITIVE NEURON NUCLEUS EXAMPLES (class = 1)
%==========================================================================

%loop through time
for t = 1:24
   
    % initialize the ground truth to be empty
    empty = zeros([1024 1024 3]);
    
    
    % loop through GT objects in the frame
    for i = 1:length(GT(t).s)
        clear mask;
        BB = GT(t).s(i).BoundingBox - [.5 .5 1 1 ];
        
        if BB(4) > BB(3)
            diff = BB(4) - BB(3);
            xshift = round(diff/2);
            BB(1) = BB(1) - xshift;
            BB(3) = BB(4);
        else
            diff = BB(3) - BB(4);
            yshift = round(diff/2);
            BB(2) = BB(2) - yshift;
            BB(4) = BB(3);
        end
        
        % find the limits of the image we want to extract for this nucleus
        y1 = BB(1);
        y2 = BB(1) + BB(3);
        x1 = BB(2);
        x2 = BB(2) + BB(4);
        
        %BORDER = max(floor([BB(3)*.3, BB(4)*.3]));
        BORDER = max(round([BB(3)/2 BB(4)/2]));
        
        
        y1 = max(1, y1 - BORDER);
        y2 = min(1024, y2 + BORDER);
        x1 = max(1, x1 - BORDER);
        x2 = min(1024, x2 + BORDER);
        
        I = GT(t).Image(x1:x2, y1:y2);
             %I = imnormalize('image', I);
             %I = adapthisteq(I);
             %I = imadjust(I);
        L = zeros(1024,1024);
        L(GT(t).s(i).PixelIdxList) = 1;
        M = L(x1:x2,y1:y2);
        s  = regionprops(M, 'orientation');
        I = imrotate(I,-s.Orientation, 'bicubic', 'crop');
        
        I = imcrop(I, [round(BORDER/2) round(BORDER/2) BB(3)+ BORDER BB(4)+BORDER]);
        
        % resize the images to a default size for learning (?)
        I = imresize(I, IMSIZE);

        % write files to the database
        nstr = number_into_string(ex_num, 10000);
        
%         imshow(I); set(gca, 'Position', [0 0 1 1]);
%         pause(.5);
%         refresh;
                
%         I1 = imrotate(I, -90);
%         train_file1 = [pos_train_path 'nucleus' nstr '.png'];
%         imwrite(I1, train_file1, 'PNG');
%         disp(['wrote  ' train_file1 ]); 
        
        I2  = I;
        test_file1 = [pos_test_path 'nucleus' nstr '.png'];
        imwrite(I2, test_file1, 'PNG');
        disp(['wrote  ' test_file1 ]); 
        
        %ex_num = ex_num + 1;
        %nstr = number_into_string(ex_num, 10000);
        
        I3 = imrotate(I, 180);
        train_file2 = [pos_train_path 'nucleus' nstr '.png'];
        imwrite(I3, train_file2, 'PNG');
        disp(['wrote  ' train_file2 ]); 
        
%         I4  = imrotate(I, 90);
%         test_file2 = [pos_test_path 'nucleus' nstr '.png'];
%         imwrite(I4, test_file2, 'PNG');
%         disp(['wrote  ' test_file2 ]); 

        ex_num = ex_num + 1;
    end
end



% %==========================================================================
% %  CREATE NEGATIVE NEURON NUCLEUS EXAMPLES (class = 0)
% %==========================================================================
% % load the correct time step (1 through 24)
% N = 6000;
% n = 1; 
% 
% W_MU = 18.4271;
% W_STD = 5.7483;
% W_VAR = sqrt(W_STD);
% LOW_VARIANCE_THRESH = .0001; %.005;         % minimum variance of a typical negative example
% 
% while n <= N
% 
% 
% 
%     % 1. Randomly select an image from the ground truth
%     t = ceil(length(GT) * rand(1));
% 
%     % 2. Load the image
%     I = GT(t).Image; 
%     Isize = size(I);
%     
% 
%     % 3. Randomly sample an EXAMPLE 
%     W = round(gsamp(W_MU, W_VAR, 1));
%     H = W;
%     X = ceil(  (Isize(2) - W)  *rand(1));
%     Y = ceil(  (Isize(1) - H) * rand(1));
% 
%     rect = [X Y W-1 H-1];
%     EXAMPLE = imcrop(I, rect);
% 
%     if var(EXAMPLE) < LOW_VARIANCE_THRESH
%         if rand(1) < .99
%             %disp('too little variance, reselecting.');
%             continue;
%         else
%             disp('little variance, but taking it anyway.');
%         end
%     end
%     
%     % 4. Check to make sure we don't overlap an annotation too much
%     x1 = rect(1);
%     y1 = rect(2);
%     x2 = rect(1) + rect(3);
%     y2 = rect(2) + rect(4);
% 
%     box1 = [x1 y1 x2 y2];
% 
%     %keyboard;
%     overlapsGT = 0;
%     for i = 1:length(GT(t).s)
%         box2(1) = GT(t).s(i).BoundingBox(1);
%         box2(2) = GT(t).s(i).BoundingBox(2);
%         box2(3) = GT(t).s(i).BoundingBox(1) + GT(t).s(i).BoundingBox(3);
%         box2(4) = GT(t).s(i).BoundingBox(2) + GT(t).s(i).BoundingBox(4);
% 
%         coverageofbox1 = overlap(box1, box2) / (W*H);
% 
%         if coverageofbox1 > .5
%             overlapsGT = 1;
%             break;
%         end
%     end
% 
% %     if ~overlapsGT
% %         EXAMPLE = imresize(EXAMPLE, IMSIZE);
% %         nstr = number_into_string(n, 10000);
% %         train_file1 = [neg_train_path 'non' nstr '.png'];
% %         imwrite(EXAMPLE, train_file1, 'PNG');
% %         disp(['wrote  ' train_file1 ]); 
% %         n = n + 1;
% %     else
% %         disp('oops this one overlaps a GT');
% %     end
%     
%     if ~overlapsGT
%         EXAMPLE = imresize(EXAMPLE, IMSIZE);
%         nstr = number_into_string(n, 10000);
%         test_file1 = [neg_test_path 'non' nstr '.png'];
%         imwrite(EXAMPLE, test_file1, 'PNG');
%         disp(['wrote  ' test_file1 ]); 
%         n = n + 1;
%     else
%         disp('oops this one overlaps a GT');
%     end
%     
% end



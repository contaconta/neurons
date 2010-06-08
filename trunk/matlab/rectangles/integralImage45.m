function II = integralImage45(I)

I = single(I);


I = padarray(I, [0 2], 'pre');
I = padarray(I, [2 0], 'pre'); 
I = padarray(I, [0 2], 'post');


IMSIZE = size(I);

II = zeros(size(I));

for r = 3:size(I,1)
    for c = 2:size(I,2)-1
        II(r,c) = II(r-1,c-1) + II(r-1,c+1) - II(r-2,c) + I(r,c) + I(r-1,c);
        
    end
end

II = II(2:size(II,1), 2:size(II,2)-1);

% DIAG = zeros(size(I)); II = zeros(size(I));
% 
% % SCAN DOWN to build DIAG
% c0 = 2;
% for r0=1:size(I,1)-1
%     r = r0;
%     c = c0;
%     % make the diag 
%     while (r > 0) && (c <= IMSIZE(2));
%                 
%         DIAG(r,c) = I(r,c) + DIAG(r+1,c-1);
%         r = r - 1;
%         c = c + 1;
%     end    
% end
% 
% % SCAN RIGHT to build DIAG
% r0 = size(I,1)-1;
% 
% for c0 = 2:size(I,2)
%     r = r0;
%     c = c0;
%     % make the diag
%     while (r > 0) && (c <= IMSIZE(2));     
%         DIAG(r,c) = I(r,c) + DIAG(r+1,c-1);
%         r = r - 1;
%         c = c + 1;   
%     end    
% end
%     
% %% do the top row scan!
% II(1,1) = I(1,1);
% II(1,2) = II(1) + DIAG(1,2);
% for c = 3:size(I,2)
%     II(1,c) = II(1,c-2) + DIAG(1,c-1);
% end
% 
% %% scan the left border
% II(:,1) = DIAG(:,1);
% 
% 
% %% scan the rest of the image
% for r = 2:size(I,1)-1
%     for c = 2:size(I,2)        
%         II(r,c) = II(r-1, c-1) + DIAG(r,c-1) + DIAG(r+1,c-1);
%     end
% end
% 
% %% scan the bottom border
% II(size(I,1),1) = I(size(I,1),1);
% for c = 2:size(I,2) 
%     II(size(II,1),c) = II(r-1, c-1) + DIAG(r,c-1);
% end
% 
% %keyboard;
%     
% % for r = 2:size(I)-1
% %     for c = 2:size(I)-1
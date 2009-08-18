% %vj_train_settings;
% versioninfo;
% tic; disp('...collecting and processing the TRAIN data set.');
% TRAIN = vj_collect_data(train1, train0, 'size', IMSIZE, 'normalize', NORM, 'data_limit', [TRAIN_POS TRAIN_NEG]);toc;
% 
% 
% %  compute spedges for each training example
% 
% %angles = [0:45:360-45];  sigma = 2;
% angles = [0:30:360-30];  sigma = 2;
% 
% tic;
% for i = 1:find([TRAIN.class] == 1, 1, 'last')
%     
%     TRAIN(i).spedge = spedges(TRAIN(i).Image, angles, sigma);
%     
% end
% toc
% 
% tic;
% for i = find([TRAIN.class] == 0, 1,  'first'):length(TRAIN)
%     
%     TRAIN(i).spedge = spedges(TRAIN(i).Image, angles, sigma);
%     
% end
% toc
% 
% POS = TRAIN([TRAIN.class] == 1);
% NEG = TRAIN([TRAIN.class] == 0);
% 

% now find the descriptor for a given point in the image

for r = 12:22  
    for c = 3:21
        
        disp(['r = ' num2str(r) ' c = ' num2str(c) ]);
        clear pos neg;
        for i = 1:length(POS)
            pos(i,:) = POS(i).spedge.spedges(:,r,c)';
        end

        for i = 1:length(NEG)
            neg(i,:) = NEG(i).spedge.spedges(:,r,c)';
        end
        
        fischers_linear_discriminant(pos, neg)
        title(['r = ' num2str(r) ' c = ' num2str(c) ]);
        

        keyboard;
    end

end




        
% r = 7;  c = 6;
% 
% for i = 1:length(POS)
%     pos(i,:) = POS(i).spedge.spedges(:,r,c)';
% end
% 
% for i = 1:length(NEG)
%     neg(i,:) = NEG(i).spedge.spedges(:,r,c)';
% end
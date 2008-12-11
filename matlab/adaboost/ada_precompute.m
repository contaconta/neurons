function SET = ada_precompute(SET, LEARNERS, WEAK, FILES, filename)


%%  allocate the bigmatrix to store all precomputed features
SET.responses = bigmatrix( length(SET.class), size(WEAK.list,1), 'filename', filename, 'memory', FILES.memory, 'precision', 'single');


%% precompute all weak learner responses
%  precompute responses of each feature of each weak learner type over the
%  entire data set

for l = 1:length(WEAK.learners)
    switch WEAK.learners{l}{1}
    
        case 'haar'
            %% haar like weak learners - for haars it is faster to compute the
            %  haar response of a singe feature over all training examples
            disp('...computing haar wavelet responses');

            % collect all the vectorized integral images into IIs
            for i = 1:length(SET.class)
                II = integral_image(SET.Images(:,:,i));
                IIs(:,i) = II(:);
            end

            W = wristwatch('start', 'end', length(WEAK.learners{l}{3}), 'every', 10000, 'text', '    ...precomputed haar feature ');
            block = round(FILES.memory / (length(SET.class)*4));        % block = # columns fit into memory lim
            R = zeros(length(SET.class), block );                       % R temporary feature response storage
            j = 1;                                                      % feature index in current block
            
            % loop through features, compute response of feature f to all
            % examples, store as columns
            for f = 1:length(WEAK.learners{l}{3})
                R(:,j) = ada_haar_response(WEAK.haars(f).hinds, WEAK.haars(f).hvals, IIs);

                if mod(f,block) == 0
                    disp(['    ...writing to ' SET.responses.filename]);
                    rows = 1:length(SET.class);
                    cols = WEAK.learners{1}{3}(f-block+1:f);
                    SET.responses.storeBlock(R,rows,cols);
                    j = 0;
                end
                W = wristwatch(W, 'update', f);
                j = j + 1;
            end
            
            % store the last columns
            disp(['    ...writing to ' SET.responses.filename]);
            cols = WEAK.learners{1}{3}(f-j+2:f);
            %A = R(:,1:j-1);
            SET.responses.storeBlock(R(:,1:j-1),rows,cols);
            
            clear R;

        %% spedge weak learners
        %  unlike haars, spedges are faster to compute by looping through
        %  the examples and computing all spedges for each example.
        case 'spedge'
            disp('...computing spedge repsonses');
            
            block = round(FILES.memory / (length(WEAK.learners{l}{3})*4)); 
            W = wristwatch('start', 'end', length(SET.class), 'every', 200, 'text', '    ...precomputed spedge for example ');
            R = zeros(block, length(WEAK.learners{l}{3}));
            j = 1;

            % loop through examples, compute all spedge repsonses for
            % example i, store as rows
            for i = 1:length(SET.class)
            
                sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).sigma);
                R(j,:) = sp.spedges(:);
                
                if mod(i,block) == 0
                    disp(['    ...writing to ' SET.responses.filename]);
                    rows = i-block+1:i;
                    cols = WEAK.learners{l}{3}(:);
                    SET.responses.storeBlock(R, rows, cols);
                    j = 0;
                end
                W = wristwatch(W, 'update', i);
                j = j + 1;
            end
            
            % store the last rows
            disp(['    ...writing to ' SET.responses.filename]);
            rows = i-j+2:i;
            SET.responses.storeBlock(R(1:j-1,:), rows, cols);
            clear R;
    end
end










% % loop through each sample
% for s = 1:length(SET.class)
% 
%     
%     
%     R = zeros(block, size(WEAK.list,1) );
%     i = 1;
% 
%     % compute the set of features associated with each type of learner
%     
%     for l = 1:length(WEAK.learners)
%         
%         
%         
%         switch WEAK.learners{l}{1}
%             
%             % if haar wavelets are used, compute them and store them
%             case 'haar'
%                 II = integral_image(SET.Images(:,:,s));
%                 
%                 hinds = {WEAK.haars(WEAK.learners{l}{3}).hinds};
%                 hvals = {WEAK.haars(WEAK.learners{l}{3}).hvals};
%                 
%                 R(i,[WEAK.learners{l}{3}]) = ada_haar_response(hinds, hvals, II(:));
%                 
%                 %for j = 1:length(WEAK.learners{l}{3})
%                 %    R(i,[WEAK.learners{l}{3}]) = ada_haar_response(WEAK.haars(j).hinds, WEAK.haars(j).hvals, II);
%                 %end
%                 
%                 
%             case 'spedge'
%                 
%         end
%         
%     end
%     
%     if mod(s,block) == 0
%         disp('writing...');
%         cols = s-block+1:s;
%         SET.responses.storeCols(R,cols);
%         i = 0;
%     end
%     W = wristwatch(W, 'update', s);
%     
%     i = i + 1;
%     
% end



% for l = 1:length(WEAK.learners)
%     switch WEAK.learners{l}{1}
%         case 'haar'
%             disp('computing haar');
% 
%             % get all the integral images
%             for i = 1:length(SET.class)
%                 II = integral_image(SET.Images(:,:,i));
%                 IIs(:,i) = II(:);
%             end
% 
%             W = wristwatch('start', 'end', length(WEAK.learners{l}{3}), 'every', 10000, 'text', '    ...precomputed haar ');
%             block = round(FILES.memory / (length(SET.class)*4)); 
%             %R = zeros(block,length(SET.class));
%             R = zeros(length(SET.class), block);
%             j = 1;
%             
%             
%             for i = 1:length(WEAK.learners{l}{3})
%                 % compute the haar response of learner i to each sample
%                 %R(j,:) = ada_haar_response(WEAK.haars(i).hinds, WEAK.haars(i).hvals, IIs);
%                 R(:,j) = ada_haar_response(WEAK.haars(i).hinds, WEAK.haars(i).hvals, IIs);
% 
%                 if mod(i,block) == 0
%                     disp('writing...');
% %                     rows = WEAK.learners{1}{3}(i-block+1:i);
% %                     SET.responses.storeRows(R,rows);
%                     cols = WEAK.learners{1}{3}(i-block+1:i);
%                     SET.responses.storeCols(R,cols);
%                     j = 0;
%                 end
%                 W = wristwatch(W, 'update', i);
%                 j = j + 1;
%             end
% 
%             clear R;
% 
% %         case 'spedge'
% %             disp('computing spedges');
% %             
% %             %block = round(FILES.memory / (length(WEAK.learners{l}{3})*4)); 
% %             
% %             for i = 1:length(SET.class)
% %             
% %                 RESPONSES = SET.responses.getCols(i);
% %                 sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).sigma);                    
% %                 R = sp.spedges(:);
% %                 RESPONSES(WEAK.learners{l}{3}) = R;
% %                 SET.responses.storeCols(RESPONSES,i);
% %             end
% %             clear R;
%     end
% 
% end

function SET = ada_precompute(SET, LEARNERS, WEAK, FILES, filename)


%%  allocate the bigmatrix to store all precomputed features
SET.responses = bigmatrix( length(SET.class), length(WEAK.learners), 'filename', filename, 'memory', FILES.memory, 'precision', FILES.precision);


%% precompute all weak learner responses
%  precompute responses of each feature of each weak learner type over the
%  entire data set

for l = 1:length(LEARNERS)
    switch LEARNERS(l).feature_type
    
    
    case 'intmean'
        %% intmean weak learners
        %  so few intmean features means we don't have to worry about storing blocks
            
            % create a list of spedge feature indexes
            f_list = []; 
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'intmean')
                    f_list = [f_list f];
                end
            end
            

            W = wristwatch('start', 'end', length(SET.class), 'every', 2000, 'text', '    ...precomputed intmean for example ');
            % loop through examples, compute all spedge repsonses for
            % example i, store as rows
            
            for f = f_list
                for i = 1:length(SET.class)
                    I = SET.Images(:,:,i);
                    R(i) = ada_intmean_response(I); 
                end
                SET.responses.storeCols(R', f);
                W = wristwatch(W, 'update', i);
            end
            
            clear R;
            
    %% intvar weak learners
    %  so few intvar features means we don't have to worry about storing blocks
    case 'intvar'
            
            % create a list of spedge feature indexes
            f_list = []; 
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'intvar')
                    f_list = [f_list f];
                end
            end
            

            W = wristwatch('start', 'end', length(SET.class), 'every', 2000, 'text', '    ...precomputed intvar for example ');
            % loop through examples, compute all spedge repsonses for
            % example i, store as rows
            
            for f = f_list
                for i = 1:length(SET.class)
                    I = SET.Images(:,:,i);
                    R(i) = ada_intvar_response(I);
                end
                SET.responses.storeCols(R', f);
                W = wristwatch(W, 'update', i);
            end
            
            clear R;
    
        
        %% haar like weak learners - for haars it is faster to compute the
        %  haar response of a single feature over all training examples
        case 'haar'
            

            % collect all the vectorized integral images into IIs
            for i = 1:length(SET.class)
                II = integral_image(SET.Images(:,:,i));
                IIs(:,i) = II(:);
            end
            
            % get a list of haar feature indexes
            num_haars = 0;  f_haars = [];
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'haar')
                    num_haars = num_haars + 1;
                    f_haars = [f_haars f];
                end
            end

            W = wristwatch('start', 'end', num_haars, 'every', 10000, 'text', '    ...precomputed haar feature ');
            block = round(FILES.memory / (length(SET.class)*SET.responses.bytes));        % block = # columns fit into memory lim
            R = zeros(length(SET.class), block );                       % R temporary feature response storage
            j = 1;                                                      % feature index in current block
            f_list = [];
            
            % loop through features, compute response of feature f to all
            % examples, store as columns
            
            for f = f_haars
                    R(:,j) = ada_haar_response(WEAK.learners{f}.hinds, WEAK.learners{f}.hvals, IIs);
                    f_list = [f_list f];

                    if mod(length(f_list),block) == 0
                        disp(['    ...writing to ' SET.responses.filename]);
                        rows = 1:length(SET.class);
                        %cols = f-block+1:f;
                        cols = f_list;
                        SET.responses.storeBlock(R,rows,cols);
                        j = 0;
                        f_list = [];
                    end
                    W = wristwatch(W, 'update', f);
                    j = j + 1;
            end
            
            % store the last columns
            disp(['    ...writing to ' SET.responses.filename]);
            cols = f_list;
            SET.responses.storeBlock(R(:,1:j-1),rows,cols);
            
            clear R;

        %% spedge weak learners
        %  unlike haars, spedges are faster to compute by looping through
        %  the examples and computing all spedges for each example.
        case 'spedge'
            
            % create a list of spedge feature indexes
            f_list = []; 
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'spedge')
                    f_list = [f_list f];
                end
            end
            
            block = min(length(SET.class), round(FILES.memory / (length(f_list)*SET.responses.bytes))); 
            W = wristwatch('start', 'end', length(SET.class), 'every', 200, 'text', '    ...precomputed spedge for example ');
            R = zeros(block, length(f_list));
            j = 1;
            
            

            % loop through examples, compute all spedge repsonses for
            % example i, store as rows
            for i = 1:length(SET.class)
            
                sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).sigma);
                R(j,:) = sp.spedges(:);
                
                if mod(i,block) == 0
                    disp(['    ...writing to ' SET.responses.filename]);
                    rows = i-block+1:i;
                    cols = f_list;
                    SET.responses.storeBlock(R, rows, cols);
                    j = 0;
                end
                W = wristwatch(W, 'update', i);
                j = j + 1;
            end
            
            if j ~= 1
                % store the last rows, if we have some left over
                disp(['    ...writing to ' SET.responses.filename]);
                rows = i-j+2:i;
                SET.responses.storeBlock(R(1:j-1,:), rows, cols);
                clear R;
            end
    
    
    
        %% spdiff weak learners
        %  unlike haars, spdiff are faster to compute by looping through
        %  the examples and computing all spedges for each example.
        case 'spdiff'
            
            % create a list of spdiff feature indexes
            f_list = []; 
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'spdiff')
                    f_list = [f_list f];
                end
            end
            
            block = min(length(SET.class), round(FILES.memory / (length(f_list)*SET.responses.bytes))); 
            W = wristwatch('start', 'end', length(SET.class), 'every', 200, 'text', '    ...precomputed spdiff for example ');
            R = zeros(block, length(f_list));
            j = 1;
            
            % loop through examples, compute all spdiff repsonses for
            % example i, store as rows
            for i = 1:length(SET.class)
            
                sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).sigma);
                
                for f = f_list
                    angle1_ind = find(LEARNERS(l).angles == WEAK.learners{f}.angle1,1);
                    angle2_ind = find(LEARNERS(l).angles == WEAK.learners{f}.angle2,1);
                    sigma_ind = find(LEARNERS(l).sigma == WEAK.learners{f}.sigma,1);
                    R(j,f) = sp.spedges(angle1_ind, sigma_ind, WEAK.learners{f}.row, WEAK.learners{f}.col) - sp.spedges(angle2_ind, sigma_ind, WEAK.learners{f}.row, WEAK.learners{f}.col);
                end
%                 
%                 R(j,:) = sp.spedges(:);
                
                if mod(i,block) == 0
                    disp(['    ...writing to ' SET.responses.filename]);
                    rows = i-block+1:i;
                    cols = f_list;
                    SET.responses.storeBlock(R, rows, cols);
                    j = 0;
                end
                W = wristwatch(W, 'update', i);
                j = j + 1;
            end
            
            if j ~= 1
                % store the last rows, if we have some left over
                disp(['    ...writing to ' SET.responses.filename]);
                rows = i-j+2:i;
                SET.responses.storeBlock(R(1:j-1,:), rows, cols);
                clear R;
            end
    
            
            
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

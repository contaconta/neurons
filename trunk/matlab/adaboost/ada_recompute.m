function SET = ada_recompute(SET, LEARNERS, WEAK, FILES)


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
            % loop through features, compute intmean responses for each 
            % example, store as a column
            
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
            % loop through features, compute intvar responses for each 
            % example, store as a column

            
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
            
            % loop through examples, compute all spedge responses for
            % example i, store as rows
            for i = 1:length(SET.class)
            
                sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).stride, LEARNERS(l).edge_methods);
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
            
            % ordered lists of the indexes of the flattened spedge tensors
            % which we will subtract form each other to create the spdiff
            ang1subs = []; ang2subs = [];
            for f = f_list
                ang1subs = [ang1subs  WEAK.learners{f}.vec_index1];
                ang2subs = [ang2subs  WEAK.learners{f}.vec_index2];
            end

            block = min(length(SET.class), round(FILES.memory / (length(f_list)*SET.responses.bytes))); 
            W = wristwatch('start', 'end', length(SET.class), 'every', 200, 'text', '    ...precomputed spdiff for example ');
            R = zeros(block, length(f_list));
            j = 1;
            
            % loop through examples, compute all spdiff repsonses for
            % example i, store as rows
            for i = 1:length(SET.class)
            
                sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).stride, LEARNERS(l).edge_methods);
                
                SP = sp.spedges(:);
                
                R(j,:) = SP(ang1subs) - SP(ang2subs);
                
%                 % DEBUGGING - make sure my speed up stuff is computing
%                 % features correctly.
%                 if i  <  5
%                     for f = 1:length(f_list)
%                         angle1_ind = find(LEARNERS(l).angles == WEAK.learners{f_list(f)}.angle1,1);
%                         angle2_ind = find(LEARNERS(l).angles == WEAK.learners{f_list(f)}.angle2,1);
%                         sigma_ind = find(LEARNERS(l).sigma == WEAK.learners{f_list(f)}.sigma,1);
%                         Rlong(j,f) = sp.spedges(angle1_ind, sigma_ind, WEAK.learners{f_list(f)}.row, WEAK.learners{f_list(f)}.col) - sp.spedges(angle2_ind, sigma_ind, WEAK.learners{f_list(f)}.row, WEAK.learners{f_list(f)}.col);
%                     
%                         Rind(j,f) = ada_spdiff_response(WEAK.learners{f_list(f)}.angle1,WEAK.learners{f_list(f)}.angle2,WEAK.learners{f_list(f)}.sigma,WEAK.learners{f_list(f)}.row,WEAK.learners{f_list(f)}.col,SET.Images(:,:,i));
%                     end
%                     keyboard;
%                 end

                
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
    
            
            
        case 'hog'
        %% hog weak learners
        %  so few hog features means we don't have to worry about storing blocks
            
            % create a list of hog feature indexes
            f_list = []; 
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'hog')
                    f_list = [f_list f];
                end
            end
            

            W = wristwatch('start', 'end', length(SET.class), 'every', 500, 'text', '    ...precomputed hog for example ');
            % loop through all examples, compute all hog feature responses
            % for example i, store as a block
            
            for i = 1:length(SET.class)
                I = SET.Images(:,:,i);
                f = HoG(I, 'orientationbins', LEARNERS(l).bins, 'cellsize', LEARNERS(l).cellsize, 'blocksize', LEARNERS(l).blocksize);
                f = f(:);
                R(i,:) = f; 
                W = wristwatch(W, 'update', i);
            end
            
            rows = 1:length(SET.class);
            cols = f_list;
            
            SET.responses.storeBlock(R, rows, cols);
            
            clear R;
            
            
            
            
            
            
    end    
end












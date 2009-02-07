function SET = ada_recompute(SET, LEARNERS, WEAK, FILES)


%% precompute all weak learner responses
%  precompute responses of each feature of each weak learner type over the
%  entire data set

for l = 1:length(LEARNERS)
    switch LEARNERS(l).feature_type
    
        case 'haar'
            %% haar like weak learners - for haars it is faster to compute the
            %  haar response of a single feature over all training examples

            % collect all the vectorized integral images into IIs
            for i = 1:length(SET.class)
                II = integral_image(SET.Images(:,:,i));
                IIs(:,i) = II(:);
            end
            
            % compute the number of haar features
            num_haars = 0;
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'haar')
                    num_haars = num_haars + 1;
                end
            end

            W = wristwatch('start', 'end', num_haars, 'every', 10000, 'text', '    ...precomputed haar feature ');
            block = round(FILES.memory / (length(SET.class)*SET.responses.bytes));        % block = # columns fit into memory lim
            R = zeros(length(SET.class), block );                       % R temporary feature response storage
            j = 1;                                                      % feature index in current block
            f_list = [];
            
            % loop through features, compute response of feature f to all
            % examples, store as columns
            
            for f = 1:length(WEAK.learners)
                if strcmp(WEAK.learners{f}.type, 'haar')
                    R(:,j) = ada_haar_response(WEAK.learners{f}.hinds, WEAK.learners{f}.hvals, IIs);
                    f_list = [f_list f];

                    if mod(f,block) == 0
                        disp(['    ...writing to ' SET.responses.filename]);
                        rows = 1:length(SET.class);
                        cols = f_list;
                        SET.responses.storeBlock(R,rows,cols);
                        j = 0;
                        f_list = [];
                    end
                    W = wristwatch(W, 'update', f);
                    j = j + 1;
                end
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
    end
end

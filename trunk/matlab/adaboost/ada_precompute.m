function SET = ada_precompute(SET, LEARNERS, WEAK, FILES, filename)



%%  allocate the bigmatrix to store all precomputed features

SET.responses = bigmatrix( length(SET.class), size(WEAK.list,1), 'filename', filename, 'memory', FILES.memory, 'precision', 'single');
       

for l = 1:length(WEAK.learners)
    switch WEAK.learners{l}{1}
        case 'haar'
            disp('computing haar');

            % get all the integral images
            for i = 1:length(SET.class)
                II = integral_image(SET.Images(:,:,i));
                IIs(:,i) = II(:);
            end

            W = wristwatch('start', 'end', length(WEAK.learners{l}{3}), 'every', 10000, 'text', '    ...precomputed haar ');
            block = round(FILES.memory / (length(SET.class)*4)); 
            %R = zeros(block,length(SET.class));
            R = zeros(length(SET.class), block);
            j = 1;
            
            
            for i = 1:length(WEAK.learners{l}{3})
                % compute the haar response of learner i to each sample
                %R(j,:) = ada_haar_response(WEAK.haars(i).hinds, WEAK.haars(i).hvals, IIs);
                R(:,j) = ada_haar_response(WEAK.haars(i).hinds, WEAK.haars(i).hvals, IIs);

                if mod(i,block) == 0
                    disp('writing...');
%                     rows = WEAK.learners{1}{3}(i-block+1:i);
%                     SET.responses.storeRows(R,rows);
                    cols = WEAK.learners{1}{3}(i-block+1:i);
                    SET.responses.storeCols(R,cols);
                    j = 0;
                end
                W = wristwatch(W, 'update', i);
                j = j + 1;
            end

            clear R;

%         case 'spedge'
%             disp('computing spedges');
%             
%             %block = round(FILES.memory / (length(WEAK.learners{l}{3})*4)); 
%             
%             for i = 1:length(SET.class)
%             
%                 RESPONSES = SET.responses.getCols(i);
%                 sp = spedges(SET.Images(:,:,i), LEARNERS(l).angles, LEARNERS(l).sigma);                    
%                 R = sp.spedges(:);
%                 RESPONSES(WEAK.learners{l}{3}) = R;
%                 SET.responses.storeCols(RESPONSES,i);
%             end
%             clear R;
    end

end

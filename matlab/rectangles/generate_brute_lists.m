

if OPT_WEIGHTS
    
    weightrange = -1:.1:1;

    % find the maxmimum rank
    RANKMAX = 4;

    brute_lists = cell(1,4);

    brute_lists{1} = [];
    b2 = cartprod(weightrange, weightrange);
    b2 = b2(b2(:,1) > 0,:);             % get rid of zeros
    b2 = b2(b2(:,2) ~= 0,:);
    b2 = b2./  repmat(b2(:,1),1,2);     % normalize the first weight
    b2 = unique(b2, 'rows');            % remove redundant weights
    brute_lists{2} = b2;
    
    b3 = cartprod(weightrange, weightrange, weightrange);
    b3 = b3(b3(:,1) > 0,:);             % get rid of zeros
    b3 = b3(b3(:,2) ~= 0,:);
    b3 = b3(b3(:,3) ~= 0,:);
    b3 = b3./  repmat(b3(:,1),1,3);     % normalize the first weight
    b3 = unique(b3, 'rows');            % remove redundant weights
    brute_lists{3} = b3;
    
    b4 = cartprod(weightrange, weightrange, weightrange, weightrange);
    b4 = b4(b4(:,1) > 0,:);             % get rid of zeros
    b4 = b4(b4(:,2) ~= 0,:);
    b4 = b4(b4(:,3) ~= 0,:);
    b4 = b4(b4(:,4) ~= 0,:);
    b4 = b4./  repmat(b4(:,1),1,4);     % normalize the first weight
    b4 = unique(b4, 'rows');            % remove redundant weights
    brute_lists{4} = b4;
    
    %clear b2 b3 b4;
    
end
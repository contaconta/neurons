

if OPT_WEIGHTS
    
    weightrange = -1:.1:1;

    % find the maxmimum rank
    RANKMAX = 2;

    brute_lists = cell(1,4);

    brute_lists{1} = [];
    brute_lists{2} = cartprod(weightrange, weightrange);
    brute_lists{3} = cartprod(weightrange, weightrange, weightrange);
    brute_lists{4} = cartprod(weightrange, weightrange, weightrange, weightrange);

end
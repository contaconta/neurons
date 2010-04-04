%function estimate_scale_parameters()

folder = '/osshare/Work/Data/LabelMe/Images/nonface/';

searchstring = [folder '*.jpg'];
d = dir(searchstring);

PYRAMID = 0;

STATS = cell([length(d) 1]);


for j  = 368:length(d)
    

    I = imread([folder d(j).name]);
    if max(size(I)) > 600
        I = impyramid(I, 'reduce');
    end

    S = 3;
    sigma0 = 1;  %.5

    % create a list of gaussian sigmas corresponding to various scales
    max_sigma = min(size(I));  %max_sigma = min(size(I)/6);
    k = 2^(1/S);
    sigma_list  = get_sigmas(sigma0, max_sigma,k);

    G = zeros(size(I,1), size(I,2), length(sigma_list));
    DoG = zeros(size(I,1), size(I,2), length(sigma_list)-1 );

    I = double(I); I = impyramid(I, 'expand');
    disp(['reading ' folder d(j).name]);

    

    norm_consts = 1  ./  (1:15/(length(sigma_list)-1):16);
    %norm_consts = ones(size(sigma_list));

    tic;
    s = 1;
    for sigma_i = sigma_list
        %disp(['sigma_i = ' num2str(sigma_i)]);

        if PYRAMID
            sigma_red = sigma_i;  red_fact = 0;
            while ( (sigma_red / 2) >= 1 ) && (red_fact <= 3)
                red_fact = red_fact + 1;
                sigma_red = sigma_red/2;
            end

            Ired = I;
            for i = 1:red_fact
                Ired = impyramid(Ired, 'reduce');
            end
            disp(['sigma_i = ' num2str(sigma_i/2), ' reduction factor = ' num2str(red_fact)  ' size = [' num2str(size(Ired)) ']' ]);
        end


        norm_const = norm_consts(s);

        if PYRAMID
            Gred = imgaussian(Ired, sigma_red);
            G(:,:,s) = imresize(Gred, [size(G,1) size(G,2)]);
        else
            disp(['sigma_i = ' num2str(sigma_i/2) ]);
            Gred = imgaussian(I, sigma_i);
            if ~isequal(size(Gred), [size(G,1) size(G,2)])
                G(:,:,s) = imresize(Gred, [size(G,1) size(G,2)]);
            else
                G(:,:,s) = Gred;
            end
        end

        if s > 1
            DoG(:,:,s-1) = abs(G(:,:,s) - G(:,:,s-1))*norm_const;
        end

        s = s + 1;
    end
    toc;

    % reverse the order of DoG and sigma_list because max always picks the
    % first in case of multiple maximal elements
    DoG = DoG(:,:, size(DoG,3):-1:1);
    sigma_list = .5*sigma_list(length(sigma_list):-1:1);

    [M maxDoG] = max(DoG, [], 3);
    scale_estimate = maxDoG;


    a1 = squeeze(median(median(DoG,1),2));
    a2 = squeeze(mean(mean(DoG,1),2));
    a3 =squeeze(max(max(DoG,[],1),[],2));
    %[a1 a2 a3];
    
    STATS{j} = [a1 a2 a3];

    % for s = 1:length(sigma_list)
    %   scale_estimate(maxDoG == s) = sigma_list(s);
    % end
    % 
    % 
    imagesc(abs(s - maxDoG)); axis image; colormap gray;
    % figure; imagesc(scale_estimate); axis image; colormap gray;
    % 
    % keyboard;


end
    






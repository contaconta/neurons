
if strcmp(RectMethod, 'Viola-Jones') || strcmp(RectMethod, 'Mixed50') || strcmp(RectMethod, 'Mixed33')|| strcmp(RectMethod, 'Asymmetric-Mix') ;
    [R,C,pre_rects,pre_cols] = generate_viola_jones_features(IMSIZE); 
elseif strcmp(RectMethod, 'VJSPECIAL')
    [R,C,pre_rects,pre_cols] = generate_viola_jones_features_special(IMSIZE);  % Rank3 has equal areas
elseif strcmp(RectMethod, 'vj24')
    [R,C,pre_rects,pre_cols] = generate_viola_jones_features(IMSIZE, 'shapes', {'vert2', 'horz2', 'checker'});
elseif strcmp(RectMethod, 'Lienhart')
    %[lien1, lien2, lien3, lien4,] = generate_lienhart_features(IMSIZE, NORM);
    disp('...loading lienhart features from the disk');
    if ~exist('pre_rects', 'var')
        switch NORM
            case 'NONORM'
                load lienhart_featuresNONORM.mat;
            case 'ANORM'
                load lienhart_featuresANORM.mat;
            case 'DNORM'
                load lienhart_featuresDNORM.mat;
        end
        pre_rects = lien1; clear lien1;
        pre_cols = lien2; clear lien2;
        pre_areas = lien3; clear lien3;
        pre_types = lien4; clear lien4;
    end
elseif strcmp(RectMethod, 'LienhartNO3')
    %[lien1, lien2, lien3, lien4,] = generate_lienhartNO3_features(IMSIZE, NORM);
    disp('...loading lienhartNO3 features from the disk');
    if ~exist('pre_rects', 'var')
        switch NORM
            case 'NONORM'
                load lienhartNO3_featuresNONORM.mat;
            otherwise 
                disp('problems!!');
                keyboard;
        end
    end
    pre_rects = lien1; clear lien1; 
    pre_cols = lien2; clear lien2;
    pre_areas = lien3; clear lien3;
    pre_types = lien4; clear lien4;
end

clear Rc C;
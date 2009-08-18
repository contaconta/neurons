
angles = LEARNERS(2).angles;
sigmas = LEARNERS(2).sigma;


I = TRAIN.Images(:,:,123);


sp = spedges(I, angles,sigmas);


for a = angles;
    for s = sigmas;
        for r = 1:24;
            for c = 1:24;
                
                sp1 = single_spedge(a,s,r,c,I);
                
                if sp1 ~= sp.spedges(find(angles == a), find(sigmas == s), r,c)
                    disp('single_spedge does not agree with spedges');
                    keyboard;
                end
                
            end
        end
    end
end




pos_inds = find(L == 1); neg_inds = find(L == -1);

switch SE
    case 'kevin'    
        disp('...weighted sampling negative examples using KEVIN method');
        %neg_inds = weight_sample(neg_inds, W(neg_inds), N_SAMPLES);
        neg_inds = weighted_sampling_efficient(neg_inds,W(neg_inds), N_SAMPLES);
        inds = [pos_inds; neg_inds];
        Lsub = L(inds);
        Wsub = W(inds); Wsub(Lsub == -1) = Wsub(Lsub==-1) * ( sum(W(L==-1))/ sum(Wsub(Lsub==-1)));
    case 'francois'
        disp('...weighted sampling negative examples using FRANCOIS method');
        %Wtemp = W;
        %Wtemp(pos_inds) = 0;
        %neg_samps = randsample(length(Wtemp), N_SAMPLES, true, Wtemp);
        neg_samps = randsample(neg_inds, N_SAMPLES, true, W(neg_inds));        
        neg_inds = unique(neg_samps);
        Wneg = zeros(length(neg_inds),1);
        for n = 1:length(neg_inds)
            Wneg(n) = sum(neg_samps == neg_inds(n));
        end
        Wneg = Wneg .*  ( sum(W(L==-1)) / sum(Wneg));
        Wsub = [W(pos_inds); Wneg];
%         Wsub = [W(pos_inds); W(neg_inds) * ((1-sum(W(pos_inds)))/sum(W(neg_inds)))];
        inds = [pos_inds; neg_inds];
        Lsub = L(inds);
        disp(['   ' num2str(length(pos_inds)) '+ and ' num2str(length(neg_inds)) '- examples sampled.']);

end


Dsub = D(inds,:);


% % temp sanity check
% Wsub = W;
% Lsub = L;
% Dsub = D;



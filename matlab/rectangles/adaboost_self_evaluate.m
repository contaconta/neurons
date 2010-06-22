function adaboost_self_evaluate(mat_filename, NEWTlist)

fplocs = [1.1e-6:.1e-6:1e-5,1.1e-5:.1e-5:1e-4,1.1e-4:.1e-4:1e-3,1.1e-3:.1e-3:1e-2, 1.1e-2:.1e-2:1e-1, 1.1e-1:.1e-1:1];
%Tlist = [50 100 200 400 600 800 1000 1200 1400 1600 1800 2000];
%Tlist = [100 700];

[pathstr,name,ext,versn] = fileparts(mat_filename); %#ok<NASGU>


disp(' ');
disp(['--------- evaluating ' name '---------']);

% load the mat file containing the classifier
load([pathstr '/' name ext]);

% % check to see if we have enough learners
% if length(CLASSIFIER.rects) < max(Tlist)
%     disp(['Error: CLASSIFIER has only ' num2str(length(CLASSIFIER.rects)) ' learners. ' num2str(max(Tlist)) ' required. Skipping ' name ext]);
%     return;
% end

if exist('Tlist', 'var')
    OLDTlist = Tlist; %#ok<NODEF>
    Tlist = unique([OLDTlist NEWTlist]);
    OLD_TPR_list = TPR_list; %#ok<NODEF>
    TPR_list = zeros(length(Tlist), length(fplocs));
    missing_list = zeros(size(Tlist));
    for t = 1:length(Tlist)
        ind = find(OLDTlist == Tlist(t));
        if ~isempty(ind)
            TPR_list(t,:) = OLD_TPR_list(ind,:);
        else
            missing_list(t) = 1;
        end
    end
else
    Tlist = NEWTlist;
    TPR_list = zeros(length(Tlist), length(fplocs));
    missing_list = ones(size(Tlist));
end


%keyboard;

NTestSets = 11;
L = [ ]; VALS = cell(1, length(Tlist));

if sum(missing_list) ~= 0
    for i = 1:NTestSets

        % load part i of the data set
        if strcmp(CLASSIFIER.method, 'Lienhart'); di_filename = ['45_TEST' num2str(i) '.mat'];
        else; di_filename = ['TEST' num2str(i) '.mat']; end;    disp(['...loading ' di_filename]); %#ok<NOSEM>
        load(di_filename);

        % collect the labels
        L = [L; Li]; %#ok<AGROW>

        fprintf('   classifying for T = ');
        for t = 1:length(Tlist)
            if missing_list(t) == 1
                T = Tlist(t);
                if length(CLASSIFIER.rects) >= T
                    rects = CLASSIFIER.rects(1:T);
                    thresh = CLASSIFIER.thresh(1:T);
                    alpha = CLASSIFIER.alpha(1:T);
                    cols  = CLASSIFIER.cols(1:T);
                    pol = CLASSIFIER.pol(1:T);
                    areas = CLASSIFIER.areas(1:T);
                    fprintf('%s ', num2str(T));
                    if isfield(CLASSIFIER, 'separeas')
                        separeas = CLASSIFIER.separeas(1:T);
                        weights = CLASSIFIER.weights(1:T);
                        VALSi = AdaBoostClassifyOPT_WEIGHTS_mex(rects, weights, separeas, thresh, pol, alpha, Di);
                    else
                        VALSi = AdaBoostClassifyDynamicA_mex(rects, cols, areas, thresh, pol, alpha, Di);
                    end
                    VALS{t} = [VALS{t}; VALSi];
                else
                    fprintf('!not%s ', num2str(T)); 
                end
            end
        end
        fprintf('\n');


    end

    % now that we have classification results, we need to evaluate the roc
    % and interpolate!
    NP = sum(L == 1);
    NN = sum(L == -1);
    fprintf('...evaluating the ROC for T = ');
    for t = 1:length(Tlist)
        if missing_list(t) == 1
            if length(CLASSIFIER.rects) >= Tlist(t)
                fprintf('%s ', num2str(Tlist(t)));
                [TP FP] = roc_evaluate2(VALS{t}, L);
                TPR_list(t,:) = interp1(FP/NN,TP/NP,fplocs);
            else
                fprintf('skipping%s', num2str(Tlist(t)));
            end
        end
    end
    fprintf('\n');
    
    valid = Tlist <= length(CLASSIFIER.rects);
    Tlist = Tlist(valid); %#ok<NASGU>
    TPR_list = TPR_list(valid,:); %#ok<NASGU>
    
    
    save([pathstr '/' name ext], 'CLASSIFIER', 'W', 'error', 'TPR_list', 'fplocs', 'Tlist');
    disp(['... saved results in ' pathstr '/' name ext ]);
else
    disp(['... ' name ' already evaluated for T = [' num2str(Tlist) ']']);
end
%keyboard;



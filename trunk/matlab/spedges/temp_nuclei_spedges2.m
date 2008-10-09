
%% PREPARE THE DATA
path(path, ['../nucleus_detection/']);
train_settings;
versioninfo;
tic; disp('...collecting and processing the TRAIN data set.');
TRAIN = vj_collect_data(train1, train0, 'size', IMSIZE, 'normalize', NORM, 'data_limit', [TRAIN_POS TRAIN_NEG]);toc;

%  compute spedges for each training example
angles = 0:30:360-30;  sigma = 2;

tic;
for i = 1:find([TRAIN.class] == 1, 1, 'last')
    
    TRAIN(i).spedge = spedges(TRAIN(i).Image, angles, sigma);
    
end
toc

tic;
for i = find([TRAIN.class] == 0, 1,  'first'):length(TRAIN)
    
    TRAIN(i).spedge = spedges(TRAIN(i).Image, angles, sigma);
    
end
toc

POS = TRAIN([TRAIN.class] == 1);
NEG = TRAIN([TRAIN.class] == 0);


%% COMPUTE  CLASS SEPARATION FOR EACH FEATURE

accuracy = zeros(24,24,length(angles));
BINS = 25;

for r = 2:23
    for c = 2:23
        for a = 1:length(angles)
            ang = angles(a);
            
            for i = 1:length(POS)
                p(i) = POS(i).spedge.spedges(a,r,c)';
            end

            for i = 1:length(NEG)
                n(i) = NEG(i).spedge.spedges(a,r,c)';
            end
            
            %histmin = min([min(p) min(n)]);
            %histmax = max([max(p) max(n)]);

            histmin = 0;
            histmax = 24;
            
            subplot(2,1,1), hist(p, [histmin:(histmax-histmin)/BINS:histmax]), title(['positive']);
            ymaxp = max(get(gca, 'YLim'));
            set(get(gca, 'Children'), 'FaceColor', [0 0 .8])
            set(get(gca, 'Children'), 'EdgeColor', [0 0 .8])
            subplot(2,1,2), hist(n, [histmin:(histmax-histmin)/BINS:histmax]), title(['negative']);
            ymaxn = max(get(gca, 'YLim'));
            subplot(2,1,1), set(gca, 'YLim', [0 max([ymaxp ymaxn])]);
            subplot(2,1,2), set(gca, 'YLim', [0 max([ymaxp ymaxn])]);
            set(get(gca, 'Children'), 'FaceColor', [.8 0 0])
            set(get(gca, 'Children'), 'EdgeColor', [.8 0 0])
            
            valsort = sort(unique([p;n])); acc = zeros(size(valsort)); tpr = zeros(size(acc)); fpr = zeros(size(acc));
    
            for i = 1:length(valsort)
                THRESH = valsort(i);
                C = [ p <= THRESH;   n <= THRESH];
                GT = [ ones(size(p));  zeros(size(p))];
                [acc(i) tpr(i) fpr(i)] = rocstats(C, GT, 'ACC', 'TPR', 'FPR');
            end

            bestthresh_ind = find(acc == max(acc), 1, 'first');
            BESTTHRESH = valsort(bestthresh_ind);

            subplot(2,1,1), line([BESTTHRESH BESTTHRESH], [0 max([ymaxp ymaxn])], 'Color', 'k', 'LineWidth', 2);
            title(['Positive Class  acc = ' num2str(acc(bestthresh_ind)) ' TPR = ' num2str(tpr(bestthresh_ind)) ' FPR = ' num2str(fpr(bestthresh_ind)) ]);

            subplot(2,1,2), line([BESTTHRESH BESTTHRESH], [0 max([ymaxp ymaxn])], 'Color', 'k', 'LineWidth', 2);
            disp(['r = ' num2str(r) ' c = ' num2str(c) ' ang = ' num2str(angles(a)) ' THRESH = ' num2str(BESTTHRESH) ' acc = ' num2str(acc(bestthresh_ind)) ' TPR = ' num2str(tpr(bestthresh_ind)) ' FPR = ' num2str(fpr(bestthresh_ind)) ]);
    
            pause(0.01);
            accuracy(r,c,a) = acc(bestthresh_ind);
        end
    end
end
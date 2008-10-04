function fischers_linear_discriminant(pos, neg)

% find the mean of each class
m_p = mean(pos);
m_n = mean(neg);



Sw = zeros(size(pos,2));

for i=1:1:size(pos,1)
   Sw = Sw + (pos(i,:)-m_p)'*(pos(i,:)-m_p);
end

for i=1:1:size(neg,1)
   Sw = Sw + (neg(i,:)-m_n)'*(neg(i,:)-m_n);
end

f = inv(Sw)*(m_n-m_p)';
n = sqrt(f'*f);
f = f./n;
f

p = pos*f;
n = neg*f;

if ~isnan(f)
    histmin = min([min(p) min(n)]);
    histmax = max([max(p) max(n)]);

    subplot(2,1,1), hist(p, [histmin:(histmax-histmin)/100:histmax]), title(['positive']);
    ymaxp = max(get(gca, 'YLim'));
    set(get(gca, 'Children'), 'FaceColor', [0 0 .8])
    set(get(gca, 'Children'), 'EdgeColor', [0 0 .8])
    subplot(2,1,2), hist(n, [histmin:(histmax-histmin)/100:histmax]), title(['negative']);
    ymaxn = max(get(gca, 'YLim'));
    subplot(2,1,1), set(gca, 'YLim', [0 max([ymaxp ymaxn])]);
    subplot(2,1,2), set(gca, 'YLim', [0 max([ymaxp ymaxn])]);
    set(get(gca, 'Children'), 'FaceColor', [.8 0 0])
    set(get(gca, 'Children'), 'EdgeColor', [.8 0 0])
    
    % find optimal threshold
    valsort = sort(unique([p;n])); acc = zeros(size(valsort)); tpr = zeros(size(acc)); fpr = zeros(size(acc));
    m_pf = m_p*f;  m_nf = m_n*f;
    i1 = min([m_pf m_nf]);
    i2 = max([m_pf m_nf]);
    valsort = valsort(valsort >= i1);
    valsort = valsort(valsort <= i2);
    if isempty(valsort)
        valsort = sort(unique([p;n])); acc = zeros(size(valsort)); tpr = zeros(size(acc)); fpr = zeros(size(acc));
    end
    
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
    disp(['THRESH = ' num2str(BESTTHRESH) ' acc = ' num2str(acc(bestthresh_ind)) ' TPR = ' num2str(tpr(bestthresh_ind)) ' FPR = ' num2str(fpr(bestthresh_ind)) ]);
    
    
%     low = histmin;  high = histmax; accuracy = .0001;  iterations = 0;  MAX_ITERATIONS = 20; STOP_ITERATIONS = 12;
%     while (low <= high) && (iterations < MAX_ITERATIONS)
%         iterations = iterations + 1;
%         THRESH = (low + high) /2;
%         
%         C = [ pos <= THRESH ;  neg < THRESH];
%         GT = [ ones(size(pos)) ; zeros(size(neg))];
%         [acc tpr fpr fps] = rocstats(C, GT, 'ACC', 'TPR', 'FPR', 'FPlist'); 
% 
%         
%     end

end



%figure; hist(p,100);
%figure; hist(n,100);

% histmin = min([min(p) min(n)]);
% histmax = max([max(p) max(n)]);
% 
% nn = hist(n, [histmin:.1:histmax]);
% np = hist(p, [histmin:.1:histmax]);

%bar([np; nn]', 'stacked')

%figure;
%hold on;
%plot(pos*f,0,'b+');
%plot(neg*f,0,'r*');
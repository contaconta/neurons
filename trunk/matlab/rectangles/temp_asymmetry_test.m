%load results/finished/Amix-cvlabpc47-Jun072010-051124.mat; ind = 5;
%load results/finished/Amix50-cvlabpc2-Jun072010-051841.mat;  ind = 4;
load results/Amix33-cvlabpc2-Jun072010-051736.mat; ind = 2;

load D.mat

C = CLASSIFIER;  IMSIZE = [24 24];
F = haar_featureDynamicA(D, C.rects{ind}, C.cols{ind}, C.areas{ind});


% + class is < threshold, therefore lowest theshold is most positive

if C.pol(ind) == -1    % SHOULD BE +1 
    [Fsorted, inds] = sort(F, 'ascend');   %SHOULD BE ascend to see + class
else
    [Fsorted, inds] = sort(F, 'descend');  % SHOULD BE descend to see + class
end

%figure; rect_vis_ind(zeros(IMSIZE), C.rects{ind}, C.cols{ind});



D = D(inds,:);
L = L(inds,:);

posinds = find(L == 1, 100);
figure;
for i = 1:100
    
    subplottight(10,10,i);
    I = ii2image(D(posinds(i),:), IMSIZE, 'outer');
    imshow(uint8(I));
    
    
    for j = 1:length(C.rects{ind})
        [r c] = ind2sub(IMSIZE+[1 1], C.rects{ind}{j});
        r = r - [0 0 0 0]; r(5) = r(1); r(3:4) = r(4:-1:3); r = r - .5;
        c = c - [0 0 0 0]; c(5) = c(1); c(3:4) = c(4:-1:3); c = c - .5;
        if C.cols{ind}(j) == 1
            h = line(c,r);
            set(h, 'color', [1 0 0]);
        else
            h = line(c,r);
            set(h, 'color', [0 1 0]);
        end
    end
end

[hneg, xn] = hist(Fsorted(find(L==-1)), 100); 
[hpos, xp] = hist(Fsorted(find(L==1)), 100); 
hpos =hpos/sum(hpos); hneg = hneg/sum(hneg);
figure; plot(xp,hpos,'b-', xn, hneg, 'r-'); line([C.thresh(ind) C.thresh(ind)], [0 max(hneg)], 'Color', [0 0 0]);
legend('+ class', '- class');
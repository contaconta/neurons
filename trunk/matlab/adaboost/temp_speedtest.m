function temp_speedtest(WEAK_NEW, WEAK, IIs)


j = 13400; tic;
for i = 1:95251
    f = test_faster(WEAK_NEW.haars(i).hinds, WEAK_NEW.haars(i).hvals, IIs);
end
toc;

tic;
for i = 1:95251
    f = ada_fast_haar_response(WEAK.fast(i,:), IIs);
end
toc;



% % j = 13400; tic;
% for i = 1:95251
%     new = test_faster(WEAK_NEW.learners(i).hinds, WEAK_NEW.learners(i).hvals,TRAIN(100).II);
%     old = ada_fast_haar_response(WEAK.fast(i,:), TRAIN(100).II);
%     if abs(old - new) > 0 %1e-12
%         keyboard;
%     end
% end
% toc;

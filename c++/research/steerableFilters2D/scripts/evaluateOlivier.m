close all
clear all

%% Testing
mask = '/media/data/steerableFilters2D/olivier/136/mask.png';
nPts = 1000;
maskIgnore = '/media/data/steerableFilters2D/olivier/136/mask_ignore.png';

    % '2/frangi_2.jpg', ...
    % '2/meijeering_2.jpg', ...
    % 'oof.jpg', ...

    % 5 is the good one

    % 'coords/136_order_2_radius_3_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_5_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_7_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_9_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_11_eOrdsInc_0_.png', ...

names = { ...
    'oof_4.jpg', ...
    'l1.png', ...
    'meijeering.jpg', ...
    'coords/136_order_2_radius_5_eOrdsInc_0_.png', ...
};
    % 'coords/136_order_2_radius_5_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_7_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_9_eOrdsInc_0_.png', ...
    % 'coords/136_order_2_radius_11_eOrdsInc_0_.png', ...

% names = { ...
    % 'oof.jpg', ...
    % 'oof_1.jpg', ...
    % 'oof_2.jpg', ...
    % 'oof_3.jpg', ...
    % 'oof_4.jpg', ...
    % 'oof_5.jpg', ...
    % 'oof_6.jpg', ...
    % 'oof_7.jpg', ...
    % 'oof_8.jpg', ...
    % 'oof_9.jpg', ...
    % 'oof_10.jpg', ...
% };



%% data harvesting
figure;
tpra = [];
fpra = [];
for i = 1:1:size(names,2)
    [tpr,fpr] = roc_from_images(['/media/data/steerableFilters2D/olivier/136/' ...
                        names{i}], mask, nPts, maskIgnore);
    tpra = [tpra ; tpr'];
    fpra = [fpra ; fpr'];
end

%% plot the stuff
plot(log10(fpra'), (tpra'))
% plot((fpra'), (tpra'))
title('NeuroStem Dataset')
xlabel('FPR')
ylabel('TPR')

set(findobj('Type','line'), 'LineWidth', 3);
legend(names, 'Location', 'SouthEast')
save2pdf('images/rocOlivierOOF.pdf')
save('evaluationOlivier.mat')

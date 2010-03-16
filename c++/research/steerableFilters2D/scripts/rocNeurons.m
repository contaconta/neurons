

mask = '/media/data/steerableFilters2D/neurons/n7/2/N7_2_mask.jpg';
nPts = 100;
maskIgnore = '/media/data/steerableFilters2D/neurons/n7/2/mask_ignore.jpg';

% resF = roc_from_images(['/media/data/steerableFilters2D/neurons/n7/2/' ...
                    % 'frangi_2.jpg'], mask, nPts, maskIgnore);
% resLDS = roc_from_images(['/media/data/steerableFilters2D/neurons/n7/' ...
                    % 'fisher_2.jpg'], mask, nPts, maskIgnore);
% resM = roc_from_images(['/media/data/steerableFilters2D/neurons/n7/2/' ...
                    % 'meijeering_2.jpg'], mask, nPts, maskIgnore );
% resH = roc_from_images(['/media/data/steerableFilters2D/neurons/n7/2/' ...
                    % 'l1.jpg'], mask, nPts, maskIgnore );
resLDSthin = roc_from_images(['/media/data/steerableFilters2D/neurons/n7/' ...
                    'fisher_2_thin_5.jpg'], mask, nPts, maskIgnore);


close all
% f = figure;
plot(log10(resF(:,1)),resF(:,2),'k')
hold on
plot(log10(resLDSthin(:,1)),resLDSthin(:,2),'y')
plot(log10(resLDS(:,1)),resLDS(:,2),'r')
plot(log10(resM(:,1)),resM(:,2),'b')
plot(log10(resH(:,1)),resH(:,2),'g')
legend('Frangi','LDS','Meijering','Hessian','Location','SouthEast');
set(findobj('Type','line'), 'LineWidth', 3)

BACKGROUNDCOLOR = [1 1 1];

height = 4;

rectNorm2 = zeros(height,4);
rectNorm2(:,1:2) = ones(height,2);
rectNorm2(:,3:4) = -1*ones(height,2);
figure; a = imagesc(rectNorm2); colormap gray; axis off; axis image;
set(gcf, 'Color', BACKGROUNDCOLOR);
l = line([.5 4.5 4.5 .5 .5], [.5 .5 height+.5 height+.5 .5]);
set(l, 'Color', [0 0 0], 'LineWidth', 3);
set(gca, 'Position', [0 0 1 1]);
g= text( [ones(height,1); 2*ones(height,1)], [[1:height]' ; [1:height]'] , '+1');
h = text( [3*ones(height,1); 4*ones(height,1)], [[1:height]' ; [1:height]'] , '-1');
for i = 1:length(h)
    set(g(i), 'FontSize', 30);
    set(h(i), 'Color', [1 1 1], 'FontSize', 30);
end
%text( [ones(height,1); 2*ones(height,1)], [[1:height]' ; [1:height]'] , '+1');
savetopdf('f1_haar2.pdf');
% figure; 
% freqz2(rectNorm2);
% set(gca, 'XTickLabel', {}); set(gca, 'YTickLabel', {}); set(gca, 'ZTickLabel', {});
% set(gcf, 'Color', BACKGROUNDCOLOR);
% axis([-1 1 -1 1 -10 50]);
% c = get(gca, 'Children'); axis off; 
% set(c, 'FaceColor', 'interp'); set(c, 'EdgeColor', 'none'); 
% %set(gcf, 'PaperOrientation', 'landscape');
% %set(gcf, 'Renderer', 'Painters');
% %savetopdf('f1_F_haar2.pdf');
% set(gca, 'Position', [0 0 1 1])
% print(gcf, '-dpng', '-r200', 'f1_F_haar2.png');

%keyboard;

rectNorm3 = zeros(height,6);
rectNorm3(:,1:2) = ones(height,2);
rectNorm3(:,3:4) = -2*ones(height,2);
rectNorm3(:,5:6) = ones(height,2);
figure; a = imagesc(rectNorm3); colormap gray; axis off; axis image;
set(gcf, 'Color', BACKGROUNDCOLOR);
l = line([.5 6.5 6.5 .5 .5], [.5 .5 height+.5 height+.5 .5]);
set(l, 'Color', [0 0 0], 'LineWidth', 3);
set(gca, 'Position', [0 0 1 1]);
g= text( [ones(height,1); 2*ones(height,1)], [[1:height]' ; [1:height]'] , '+1');
h = text( [3*ones(height,1); 4*ones(height,1)], [[1:height]' ; [1:height]'] , '-1');
k= text( [5*ones(height,1); 6*ones(height,1)], [[1:height]' ; [1:height]'] , '+1');
for i = 1:length(h)
    set(g(i), 'FontSize', 30);
    set(h(i), 'Color', [1 1 1], 'FontSize', 30);
    set(k(i), 'FontSize', 30);
end
savetopdf('f1_haar3.pdf');
% figure; 
% freqz2(rectNorm3);
% %set(gca, 'XTick', []); set(gca, 'YTick', []); set(gca, 'ZTick', []);
% set(gca, 'XTickLabel', {}); set(gca, 'YTickLabel', {}); set(gca, 'ZTickLabel', {});
% set(gcf, 'Color', BACKGROUNDCOLOR);
% axis([-1 1 -1 1 -10 50]);
% c = get(gca, 'Children'); axis off; 
% set(c, 'FaceColor', 'interp'); set(c, 'EdgeColor', 'none'); 
% %savetopdf('f1_F_haar3.pdf');
% set(gca, 'Position', [0 0 1 1])
% print(gcf, '-dpng', '-r200', 'f1_F_haar3.png');

rectNonNorm3 = zeros(height,6);
rectNonNorm3(:,1:2) = ones(height,2);
rectNonNorm3(:,3:4) = -1*ones(height,2);
rectNonNorm3(:,5:6) = ones(height,2);
figure; a = imagesc(rectNonNorm3); colormap gray; axis off; axis image;
set(gcf, 'Color', BACKGROUNDCOLOR);
l = line([.5 6.5 6.5 .5 .5], [.5 .5 height+.5 height+.5 .5]);
set(l, 'Color', [0 0 0], 'LineWidth', 3);
set(gca, 'Position', [0 0 1 1]);
g= text( [ones(height,1)-.4; 2*ones(height,1)-.4], [[1:height]' ; [1:height]'] , '+1/16');
h = text( [3*ones(height,1)-.3; 4*ones(height,1)-.3], [[1:height]' ; [1:height]'] , '-1/8');
k= text( [5*ones(height,1)-.4; 6*ones(height,1)-.4], [[1:height]' ; [1:height]'] , '+1/16');
for i = 1:length(h)
    set(g(i), 'FontSize', 30);
    set(h(i), 'Color', [1 1 1], 'FontSize', 30);
    set(k(i), 'FontSize', 30);
end
savetopdf('f1_haar3Norm.pdf');
% figure; 
% freqz2(rectNonNorm3);
% %set(gca, 'XTick', []); set(gca, 'YTick', []); set(gca, 'ZTick', []);
% set(gca, 'XTickLabel', {}); set(gca, 'YTickLabel', {}); set(gca, 'ZTickLabel', {});
% set(gcf, 'Color', BACKGROUNDCOLOR);
% axis([-1 1 -1 1 -10 50]);
% c = get(gca, 'Children'); axis off; 
% set(c, 'FaceColor', 'interp'); set(c, 'EdgeColor', 'none'); 
% %savetopdf('f1_F_haarNon3.pdf');
% set(gca, 'Position', [0 0 1 1])
% print(gcf, '-dpng', '-r200', 'f1_F_haar4.png');

height = 4;
rectNorm4 = ones(height,4);
rectNorm4(1:height/2,3:4) = -1*ones(height/2,2);
rectNorm4((height/2)+1:height,1:2) = -1*ones(height/2,2);
figure; a = imagesc(rectNorm4); colormap gray; axis off; axis image;
set(gcf, 'Color', BACKGROUNDCOLOR);
l = line([.5 4.5 4.5 .5 .5], [.5 .5 height+.5 height+.5 .5]);
set(l, 'Color', [0 0 0], 'LineWidth', 3);
set(gca, 'Position', [0 0 1 1]);
g= text( [ones(2,1); 2*ones(2,1)], [[1:2]' ; [1:2]'] , '+1');
h = text( [3*ones(2,1); 4*ones(2,1)], [[1:2]' ; [1:2]'] , '-1');
k= text( [3*ones(2,1); 4*ones(2,1)], [[3:4]' ; [3:4]'] , '+1');
l = text( [ones(2,1); 2*ones(2,1)], [[3:4]' ; [3:4]'] , '-1');
for i = 1:length(h)
    set(g(i), 'FontSize', 30);
    set(h(i), 'Color', [1 1 1], 'FontSize', 30);
    set(k(i), 'FontSize', 30);
    set(l(i), 'Color', [1 1 1], 'FontSize', 30);
end
savetopdf('f1_haar4.pdf');
% figure; 
% freqz2(rectNorm4);
% %set(gca, 'XTick', []); set(gca, 'YTick', []); set(gca, 'ZTick', []);
% set(gca, 'XTickLabel', {}); set(gca, 'YTickLabel', {}); set(gca, 'ZTickLabel', {});
% set(gcf, 'Color', BACKGROUNDCOLOR);
% axis([-1 1 -1 1 -10 50]);
% c = get(gca, 'Children'); axis off; 
% set(c, 'FaceColor', 'interp'); set(c, 'EdgeColor', 'none'); 
% %savetopdf('f1_F_haar4.pdf');
% set(gca, 'Position', [0 0 1 1])
% print(gcf, '-dpng', '-r200', 'f1_F_haar3Non.png');




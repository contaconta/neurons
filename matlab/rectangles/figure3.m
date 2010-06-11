% height =4;
% rectNorm3 = zeros(height,6);
% rectNorm3(:,1:2) = -1*ones(height,2);
% rectNorm3(:,3:4) = ones(height,2);
% rectNorm3(:,5:6) = -1*ones(height,2);
% figure; a = imagesc(rectNorm3); colormap gray; axis off; axis image;
% set(gcf, 'Color', BACKGROUNDCOLOR);
% l = line([.5 6.5 6.5 .5 .5], [.5 .5 height+.5 height+.5 .5]);
% set(l, 'Color', [0 0 0], 'LineWidth', 3);
% set(gca, 'Position', [0 0 1 1]);
% g= text( 1, 2.5 , '$\frac{A_2}{2}$');
% set(g,'String','$\frac{A_2}{2}$','Interpreter','latex', 'FontSize', 70,'Color', [1 1 1])
% h = text( 3.2, 2.5  , 'A_1', 'FontSize', 50);
% k= text( 5, 2.5 , '$\frac{A_2}{2}$');
% set(k,'String','$\frac{A_2}{2}$','Interpreter','latex', 'FontSize', 70,'Color', [1 1 1])
% % for i = 1:length(h)
% %     set(g(i), 'Color', [1 1 1], 'FontSize', 70);
% %     set(h(i),  'FontSize', 70);
% %     set(k(i), 'Color', [1 1 1], 'FontSize', 70);
% % end
% savetopdf('f3_haar3.pdf');



S = zeros(60,50);
S(:,1:20) = 105;
S(:,21:40) = 180;
S(:,41:60) = 105;
S = S + round(10*randn(size(S)));

figure; imshow(uint8(S));
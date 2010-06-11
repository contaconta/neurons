
% rectNorm3 = zeros(height,6);
% rectNorm3(:,1:2) = -1*ones(height,2);
% rectNorm3(:,3:4) = 1*ones(height,2);
% rectNorm3(:,5:6) = -1*ones(height,2);
% figure; a = imagesc(rectNorm3); colormap gray; axis off; axis image;
% set(gcf, 'Color', BACKGROUNDCOLOR);
% l = line([.5 6.5 6.5 .5 .5], [.5 .5 height+.5 height+.5 .5]);
% set(l, 'Color', [0 0 0], 'LineWidth', 3);
% set(gca, 'Position', [0 0 1 1]);
% savetopdf('f4_haar3.pdf');



BACKGROUNDCOLOR = [1 1 1];
figure; set(gcf, 'Color', BACKGROUNDCOLOR);
a = plot([0 -128 -255], [1 1 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
h=line([-255 -255], [0 1]);
set(h, 'Color',[0 0 0], 'LineWidth', 2)
axis([-510 510 0 1.5]);
y = line([0 0],[ 0 1.5]);
set(y, 'Color',[0 0 0]);
x = line([-510 510],[ 0 0]);
set(x, 'Color',[0 0 0]);
a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
box on
set(a, 'TickLength', [0 0]);
set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
%yblock = line([XLim(1) XLim(1)], YLim, 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
%set(yblock, 'ZData', [-1 -1]);
set(gca, 'YTick', []);
set(gca, 'YColor', [1 1 1]);
set(gca, 'XTick', []);
set(gca, 'XColor', [1 1 1]);
set(gcf, 'Position', [560 529 1200 300]);
set(gcf, 'PaperPositionMode', 'auto');
%x1 = text( -285, -.1  , '-255wh', 'FontSize', 18);
%x2 = text( 235, -.1  , '255wh', 'FontSize', 18);
x = xlabel('VJ^3 Response');
m = text(-128, .5, 'blah');
set(m,'String','$ \frac{\mu_1}{\mu_2}$','Interpreter','latex', 'FontSize', 40)
set(x, 'FontSize', 22, 'Color', [ 0 0 0]);
savetopdf('f4_flat.pdf');



BACKGROUNDCOLOR = [1 1 1];
figure; set(gcf, 'Color', BACKGROUNDCOLOR);
a = plot([-255 -128 0 128 255], [1 1 1 1 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
h1=line([-255 -255], [0 1]);
set(h1, 'Color',[0 0 0], 'LineWidth', 2)
h2=line([255 255], [0 1]);
set(h2, 'Color',[0 0 0], 'LineWidth', 2)
axis([-510 510 0 1.5]);
y = line([0 0],[ 0 1.5]);
x = line([-510 510],[ 0 0]);
set(x, 'Color',[0 0 0]);
set(y, 'Color',[0 0 0]);
a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
box on
set(a, 'TickLength', [0 0]);
set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
%yblock = line([XLim(1) XLim(1)], YLim, 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
%set(yblock, 'ZData', [-1 -1]);
set(gca, 'YTick', []);
set(gca, 'YColor', [1 1 1]);
set(gca, 'XTick', []);
set(gca, 'XColor', [1 1 1]);
set(gcf, 'Position', [560 529 1200 300]);
set(gcf, 'PaperPositionMode', 'auto');
%x1 = text( -285, -.1  , '-255wh', 'FontSize', 18);
%x2 = text( 235, -.1  , '255wh', 'FontSize', 18);
x = xlabel('VJ^3 Response');
m1 = text(-158, .5, 'blah');
set(m1,'String',' $ \frac{\mu_1}{\mu_2}$','Interpreter','latex', 'FontSize', 40)
m2 = text(100, .5, 'blah');
set(m2,'String','$ \frac{\mu_1}{\mu_2}$','Interpreter','latex', 'FontSize', 40)
set(x, 'FontSize', 22, 'Color', [0 0 0]);
savetopdf('f4_corr.pdf');




BACKGROUNDCOLOR = [1 1 1];
figure; set(gcf, 'Color', BACKGROUNDCOLOR);
a = plot([-510 -255 0], [1 1 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
h1=line([-510 -510], [0 1]);
set(h1, 'Color',[0 0 0], 'LineWidth', 2)
h2=line([0 0], [0 1]);
set(h2, 'Color',[0 0 0], 'LineWidth', 2)
axis([-510 510 0 1.5]);
y = line([0 0],[ 0 1.5]);
x = line([-510 510],[ 0 0]);
set(x, 'Color',[0 0 0]);
set(y, 'Color',[0 0 0]);
a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
box on
set(a, 'TickLength', [0 0]);
set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
%yblock = line([XLim(1) XLim(1)], YLim, 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
%set(yblock, 'ZData', [-1 -1]);
set(gca, 'YTick', []);
set(gca, 'YColor', [1 1 1]);
set(gca, 'XTick', []);
set(gca, 'XColor', [1 1 1]);
set(gcf, 'Position', [560 529 1200 300]);
set(gcf, 'PaperPositionMode', 'auto');
%x1 = text( -285, -.1  , '-255wh', 'FontSize', 18);
%x2 = text( 235, -.1  , '255wh', 'FontSize', 18);
x = xlabel('VJ^3 Response');
m1 = text(-255, .5, 'blah');
set(m1,'String',' $ \frac{\mu_1}{\mu_2}$','Interpreter','latex', 'FontSize', 40)
%m2 = text(100, .5, 'blah');
%set(m2,'String','$ \frac{\mu_1}{\mu_2}$','Interpreter','latex', 'FontSize', 40)
set(x, 'FontSize', 22, 'Color', [0 0 0]);
savetopdf('f4_anti.pdf');


% BACKGROUNDCOLOR = [1 1 1];
% figure; set(gcf, 'Color', BACKGROUNDCOLOR);
% a = plot([-255 -128 0 128 255], [1 1 1 1 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
% %a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
% h1=line([-255 -255], [0 1]);
% set(h1, 'Color',[0 0 0], 'LineWidth', 2)
% h2=line([255 255], [0 1]);
% set(h2, 'Color',[0 0 0], 'LineWidth', 2)
% axis([-510 510 0 1.5]);
% y = line([0 0],[ 0 1.5]);
% x = line([-520 510],[ 0 0]);
% set(x, 'Color',[0 0 0]);
% set(y, 'Color',[0 0 0]);
% a = gca;
% XLim = get(a, 'XLim');
% YLim = get(a, 'YLim');
% YTick = get(a, 'YTick');
% box on
% set(a, 'TickLength', [0 0]);
% set(a, 'Color', BACKGROUNDCOLOR);
% set(a, 'LineWidth', 0.001);
% %yblock = line([XLim(1) XLim(1)], YLim, 'Linewidth', 0.5, 'Color', BACKGROUNDCOLOR);
% %set(yblock, 'ZData', [-1 -1]);
% set(gca, 'YTick', []);
% set(gca, 'YColor', [1 1 1]);
% set(gca, 'XTick', []);
% set(gca, 'XColor', [1 1 1]);
% set(gcf, 'Position', [560 529 1200 300]);
% set(gcf, 'PaperPositionMode', 'auto');
% %x1 = text( -285, -.1  , '-255wh', 'FontSize', 18);
% %x2 = text( 235, -.1  , '255wh', 'FontSize', 18);
% x = xlabel('VJ^3 Response');
% % m1 = text(-158, .5, 'blah');
% % set(m1,'String','\mu_2 < \mu_1','Interpreter','latex', 'FontSize', 25)
% % m2 = text(100, .5, 'blah');
% % set(m2,'String','$\mu_2 > \mu_1$','Interpreter','latex', 'FontSize', 25)
% set(x, 'FontSize', 22, 'Color', [0 0 0]);
% savetopdf('f4_norm.pdf');



BACKGROUNDCOLOR = [1 1 1];
figure; set(gcf, 'Color', BACKGROUNDCOLOR);
a = plot([ 0 128 255], [ 1 1 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%h1=line([-255 -255], [0 1]);
%set(h1, 'Color',[0 0 0], 'LineWidth', 2)
h2=line([255 255], [0 1]);
set(h2, 'Color',[0 0 0], 'LineWidth', 2)
axis([-510 510 0 1.5]);
y = line([0 0],[ 0 1.5]);
x = line([-520 510],[ 0 0]);
set(x, 'Color',[0 0 0]);
set(y, 'Color',[0 0 0]);
a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
box on

set(a, 'TickLength', [0 0]);
set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
set(gca, 'YTick', []);
set(gca, 'YColor', [1 1 1]);
set(gca, 'XTick', []);
set(gca, 'XColor', [1 1 1]);
set(gcf, 'Position', [560 529 1200 300]);
set(gcf, 'PaperPositionMode', 'auto');
x = xlabel('VJ^3 Response');
set(x, 'FontSize', 22, 'Color', [0 0 0]);
savetopdf('f4_norm_pos.pdf');

BACKGROUNDCOLOR = [1 1 1];
figure; set(gcf, 'Color', BACKGROUNDCOLOR);
a = plot([ 0  ], [ 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%h1=line([-255 -255], [0 1]);
%set(h1, 'Color',[0 0 0], 'LineWidth', 2)
h2=line([0 0], [0 1]);
set(h2, 'Color',[0 0 0], 'LineWidth', 2)
axis([-510 510 0 1.5]);
y = line([0 0],[ 0 1.5]);
x = line([-520 510],[ 0 0]);
set(x, 'Color',[0 0 0]);
set(y, 'Color',[0 0 0]);
a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
box on
x = xlabel('VJ^3 Response');
set(a, 'TickLength', [0 0]);
set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
set(gca, 'YTick', []);
set(gca, 'YColor', [1 1 1]);
set(gca, 'XTick', []);
set(gca, 'XColor', [1 1 1]);
set(gcf, 'Position', [560 529 1200 300]);
set(gcf, 'PaperPositionMode', 'auto');
x = xlabel('VJ^3 Response');
set(x, 'FontSize', 22, 'Color', [0 0 0]);
savetopdf('f4_norm_flat.pdf');

BACKGROUNDCOLOR = [1 1 1];
figure; set(gcf, 'Color', BACKGROUNDCOLOR);
a = plot([ -255 -128 0], [ 1 1 1], 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
%a = area([0 -255], [1 1], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0 0]);
h1=line([-255 -255], [0 1]);
set(h1, 'Color',[0 0 0], 'LineWidth', 2)
%h2=line([255 255], [0 1]);
%set(h2, 'Color',[0 0 0], 'LineWidth', 2)
axis([-510 510 0 1.5]);
y = line([0 0],[ 0 1.5]);
x = line([-520 510],[ 0 0]);
set(x, 'Color',[0 0 0]);
set(y, 'Color',[0 0 0]);
a = gca;
XLim = get(a, 'XLim');
YLim = get(a, 'YLim');
YTick = get(a, 'YTick');
box on
x = xlabel('VJ^3 Response');
set(a, 'TickLength', [0 0]);
set(a, 'Color', BACKGROUNDCOLOR);
set(a, 'LineWidth', 0.001);
set(gca, 'YTick', []);
set(gca, 'YColor', [1 1 1]);
set(gca, 'XTick', []);
set(gca, 'XColor', [1 1 1]);
set(gcf, 'Position', [560 529 1200 300]);
set(gcf, 'PaperPositionMode', 'auto');
x = xlabel('VJ^3 Response');
set(x, 'FontSize', 22, 'Color', [0 0 0]);
savetopdf('f4_norm_anti.pdf');


%h2 = findobj(get(h,'Children'),'String','VJ^3');
%set(h2,'String','$\overline{VJ}^3$','Interpreter','latex')

ch = 126;
u1 = 128-ch; u2 = 128+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_cor_str.png'], 'PNG');

ch = 20;
u1 = 128-ch; u2 = 128+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_cor_med.png'], 'PNG');
%imtool(uint8(S), 'InitialMagnification', 2000 );

ch = 0;
u1 = 128-ch; u2 = 128+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_cor_wk.png'], 'PNG');


ch = -105;
u1 = 128-ch; u2 = 128+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_anti_str.png'], 'PNG');

ch = -20;
u1 = 128-ch; u2 = 128+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_anti_med.png'], 'PNG');
%imtool(uint8(S), 'InitialMagnification', 2000 );

ch = 0;
u1 = 128-ch; u2 = 128+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_anti_wk1.png'], 'PNG');

ch = 0;
u1 = 25; u2 = 25+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_anti_wk2.png'], 'PNG');

ch = 0;
u1 = 230-ch; u2 = 230+ch;
S = zeros(12,18);
S(:,1:6) = u1;
S(:,7:12) = u2;
S(:,13:18) = u1;
S = S + round(10*randn(size(S)));
imwrite(uint8(S), [pwd '/figures/f4_anti_wk3.png'], 'PNG');

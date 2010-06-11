figure;
load results/finished/VJ-cvlabpc47-May312010-000052.mat
plot_classifier_composition(CLASSIFIER);
h=legend('VJ^2', 'VJ^3', 'VJ^4', 'Location', 'SouthEast');
set(h, 'FontSize', 18);
set(gcf, 'Position', [560 529 800 420]);
x=xlabel('Boosting Stage (t)');
y=ylabel('Percent of Weak Learners (%)');
set(x, 'FontSize', 18); set(y, 'FontSize', 18); 
%set(gcf, 'PaperPositionMode',  auto);
set(gcf, 'PaperPositionMode', 'auto');
savetopdf('f2_VJ_composition.pdf');

figure;
load results/finished/VJANORM-insunrays1-Jun022010-213245.mat
plot_classifier_composition(CLASSIFIER);

h = legend('VJ^2', 'VJ^3', 'VJ^4', 'Location', 'SouthEast');
set(h, 'FontSize', 18);
h2 = findobj(get(h,'Children'),'String','VJ^3');
set(h2,'String','$\overline{VJ}^3$','Interpreter','latex')
set(gcf, 'Position', [560 529 800 420]);
x=xlabel('Boosting Stage (t)');
y=ylabel('Percent of Weak Learners (%)');
set(x, 'FontSize', 18); set(y, 'FontSize', 18); 
%set(gcf, 'PaperPositionMode',  'auto');
set(gcf, 'PaperPositionMode', 'auto');
savetopdf('f2_VJANORM_composition.pdf');
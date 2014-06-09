function plot_property(cell_list, propertyname, Sequence, cols, mode)

% mode 0 = no um conversion
% mode 1 = use um coversion

addpath export_fig

a = .0771;  % um per pixel
FONTSIZE = 10;

h1 = figure;

hold on;

for i = 1:numel(cell_list)
    id = cell_list(i);
    color = cols(id,:);
    
    
%     data = Sequence.TrackedCells(id).(propertyname);
    data = [Sequence.TrackedCells(id).TimeStep(:).(propertyname)];
    
    switch mode
        case 1  
            data = data * a;
    end
    
    
    if strcmpi(propertyname, 'DistanceTraveled');
        data = cumsum(data);
    end
    
    plot(data, 'Color', color, 'LineWidth', 2);
        
end

a1 = axis;
axis([0 97 a1(3) a1(4)]);
% set(gca, 'XTick', [0 10 20 30 40 50 60 70 80 90]);
% set(gca, 'XTickLabel', {'00:00', '01:48', '03:48', '05:48', '07:48', '09:48', '11:48', '13:48', '15:48', '17:48'});

set(gcf, 'Position', [1930 735 703 218]);
set(gcf, 'Color', [1 1 1]);
set(gca, 'box', 'off');
set(gca, 'XTick', [0 20 40 60 80]);
set(gca, 'XTickLabel', {'00:00', '03:48', '07:48','11:48', '15:48'});

set(gca, 'FontName', 'Helvetica', 'FontSize', FONTSIZE); 
set(h1, 'InvertHardcopy', 'off');



filename = sprintf('/home/ksmith/code/neurons/matlab/baselFigure/inkscape/%s.pdf', propertyname);

export_fig /home/ksmith/code/neurons/matlab/baselFigure/inkscape/temp.pdf -pdf
copyfile('/home/ksmith/code/neurons/matlab/baselFigure/inkscape/temp.pdf', filename);



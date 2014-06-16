function plot_property(cell_list, propertyname, Sequence, cols, folder, mode)

% mode 0 = no um conversion
% mode 1 = use um coversion

if ~exist(folder, 'dir')
    mkdir(folder);
end

addpath export_fig

a = .0771;  % um per pixel
FONTSIZE = 10;

h1 = figure;

hold on;

for i = 1:numel(cell_list)
    id = cell_list(i);
    color = cols(id,:);
    
    
%     data = Sequence.TrackedCells(id).(propertyname);
    ydata = [Sequence.TrackedCells(id).TimeStep(:).(propertyname)];
    xdata = [Sequence.TrackedCells(id).TimeStep.Time];
    switch mode
        case 1  
            ydata = ydata * a;
    end
    
    
    if strcmpi(propertyname, 'DistanceTraveled') 
        ydata = [ydata(1) cumsum(ydata)];
    elseif  strcmpi(propertyname, 'Speed')
        ydata = [ydata(1) ydata];
    end
    
    plot(xdata, ydata, 'Color', color, 'LineWidth', 2);
        
end

a1 = axis;
if strcmpi(propertyname,'NumberOfNeurites')
    axis([0 97 0 a1(4)]);
else
    axis([0 97 a1(3) a1(4)]);
end

% set(gca, 'XTick', [0 10 20 30 40 50 60 70 80 90]);
% set(gca, 'XTickLabel', {'00:00', '01:48', '03:48', '05:48', '07:48', '09:48', '11:48', '13:48', '15:48', '17:48'});

set(gcf, 'Position', [1930 735 703 218]);
set(gcf, 'Color', [1 1 1]);
set(gca, 'box', 'off');
set(gca, 'XTick', [0 20 40 60 80]);
set(gca, 'XTickLabel', {'00:00', '03:48', '07:48','11:48', '15:48'});

set(gca, 'FontName', 'Helvetica', 'FontSize', FONTSIZE); 
set(h1, 'InvertHardcopy', 'off');



filename = sprintf('%s%s.pdf', folder, propertyname);
temp_file = [folder 'temp.pdf'];
export_fig(temp_file, '-pdf');
copyfile(temp_file, filename);
delete(temp_file);


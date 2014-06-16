function make_csv(cell_list, measurements, Sequence, folder, mode)

% mode 0 = no um conversion
% mode 1 = use um coversion

if ~exist(folder, 'dir')
    mkdir(folder);
end
a = .0771;  % um per pixel




for m = 1:numel(measurements)
    propertyname = measurements{m};
    

    % get the time data
    tdata = 1:97;
    xdata = {};
    ydata = {};
    tstr = {};
    for t = 1:numel(tdata)
        nSec = (tdata(t) - 1) * 10.8 * 60;
        tstr{t} = datestr(nSec/86400, 'HH:MM');
    end
    
    
    for i = 1:numel(cell_list)
        id = cell_list(i);
        ydata{i} = [Sequence.TrackedCells(id).TimeStep(:).(propertyname)];
        xdata{i} = [Sequence.TrackedCells(id).TimeStep.Time];
%         switch mode
%             case 1
%                 ydata{i} = ydata{i} * a;
%         end
        
        
        if strcmpi(propertyname, 'DistanceTraveled')
            ydata{i} = [ydata{i}(1) cumsum(ydata{i})];
        elseif  strcmpi(propertyname, 'Speed')
            ydata{i} = [ydata{i}(1) ydata{i}];
        end
        
%         plot(xdata{i}, ydata{i}, 'Color', color, 'LineWidth', 2);
        
    end

    filename = sprintf('%s%s.csv', folder, propertyname);
    fid = fopen(filename,'w');
    
    % header
    fprintf(fid,'frame,time,');
    for i = 1:numel(cell_list)-1
        id = cell_list(i);
        fprintf(fid, 'cell %d,', id);
    end
    id = cell_list(end);
    fprintf(fid, 'cell %d\n', id);
    
    % body
    for t = 1:numel(tdata)
        
        fprintf(fid, '%d,%s,',t,tstr{t});
        
        for i = 1:numel(cell_list)-1
            if isempty(find(xdata{i} == t))
                fprintf(fid,',');
            else
                ind = find(xdata{i} == t);
                y = ydata{i}(ind);
                fprintf(fid, '%6.4f,', y);
            end
        end
        i = numel(cell_list);
        if isempty(find(xdata{i} == t))
        else
            ind = find(xdata{i} == t);
            y = ydata{i}(ind);
            fprintf(fid, '%6.4f', y);
        end
        fprintf(fid, '\n');
    end
    
    fprintf('wrote %s\n', filename);
    fclose(fid);
end







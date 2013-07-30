function [] = write_obj(Graph, Branches, filename)



fid = fopen(filename,'wt');
if( fid==-1 )
    error('Can''t open the file.');
    return;
end


for i = 1:numel(Graph.X)
    fprintf(fid, 'v %f %f %f %f\n', Graph.X(i), Graph.Y(i), Graph.Z(i), Graph.D(i)/2);
end

for i = 1:length(Branches)
%     if(length(Branches{i}) >=2)
%         for j = 1:length(Branches{i})-1
%             fprintf(fid, 'l %d %d\n', Branches{i}(j), Branches{i}(j+1));
%         end
%     end
    if(length(Branches{i}) >=2)
        currentString = 'l';
        for j = 1:length(Branches{i})
            currentString = [currentString ' ' int2str(Branches{i}(j))];%#ok
        end
        fprintf(fid, '%s\n', currentString);
    end
    
end

fclose(fid);
function Graph = MergeGraphs(Trees)

Graph.X = Trees{1}.X;
Graph.Y = Trees{1}.Y;
Graph.Z = Trees{1}.Z;
Graph.D = Trees{1}.D;
Graph.R = Trees{1}.R;
for i = 2:length(Trees)
    Graph.X = [Graph.X; Trees{i}.X];
    Graph.Y = [Graph.Y; Trees{i}.Y];
    Graph.Z = [Graph.Z; Trees{i}.Z];
    Graph.D = [Graph.D; Trees{i}.D];
    Graph.R = [Graph.R; Trees{i}.R];
end

dA = sparse(numel(Graph.X), numel(Graph.X));

offset = 0;
for i = 1:length(Trees)
    currentIndices = find(Trees{i}.dA);
    [I,J] = ind2sub(size(Trees{i}.dA),currentIndices);
    for k =1:length(currentIndices)
        dA(I(k)+offset, J(k)+offset) = 1;%#ok
    end
    offset = offset + numel(Trees{i}.X);
end
Graph.dA = dA;
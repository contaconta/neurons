function Branches = GetBranches(Graph)

nb_sons = zeros(1, size(Graph.dA, 1));
for k = 1:size(Graph.dA, 1)
    nb_sons(k) = sum(Graph.dA(:, k));
end

listOfTerminations = find(nb_sons == 0);
nb_terminations    = numel(listOfTerminations);

listOfBifurcations = find(nb_sons > 1);
nb_bifurcation     = numel(listOfBifurcations);

Branches = cell(nb_terminations + nb_bifurcation, 1);

for i =1:nb_terminations
    currentPoint = listOfTerminations(i);
    currentBranche = [];
    nbParents      = sum(Graph.dA(currentPoint, :));
    nbSons         = sum(Graph.dA(:, currentPoint));
    while(nbParents == 1 && nbSons <= 1)
        currentBranche = [currentPoint currentBranche ];%#ok
        currentPoint = find(Graph.dA(currentPoint, :));
        nbParents      = sum(Graph.dA(currentPoint, :));
        nbSons         = sum(Graph.dA(:, currentPoint));
    end
    currentBranche = [currentPoint  currentBranche];%#ok
    Branches{i} = currentBranche;
end


for i =1:nb_bifurcation
    currentPoint = listOfBifurcations(i);
    currentBranche = [currentPoint];
    if ~isempty(find(Graph.dA(currentPoint, :), 1))
        currentPoint = find(Graph.dA(currentPoint, :));
        nbParents      = sum(Graph.dA(currentPoint, :));
        nbSons         = sum(Graph.dA(:, currentPoint));
        while(nbParents == 1 && nbSons <= 1)
            currentBranche = [currentBranche currentPoint];%#ok
            currentPoint = find(Graph.dA(currentPoint, :));
            nbParents      = sum(Graph.dA(currentPoint, :));
            nbSons         = sum(Graph.dA(:, currentPoint));
        end
        currentBranche = [currentBranche currentPoint];%#ok
    end
    Branches{i+nb_terminations} = currentBranche;
end

if 0
    figure;
    hold on;
    for i =1:nb_terminations
       plot3(Graph.X(Branches{i}), Graph.Y(Branches{i}), Graph.Z(Branches{i}), '-b');
       pause;
    end
    for i =1:nb_bifurcation
        plot3(Graph.X(Branches{i+nb_terminations}), Graph.Y(Branches{i+nb_terminations}), Graph.Z(Branches{i+nb_terminations}), '-r');
        pause;

    end

    keyboard;
end
function [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites(Probs, Soma, Centroid, Filaments)

% [parents, neuriteId, branchesLeafs] = breakSkeletonIntoNeurites ...
%            (Probs, Soma, Centroid, Filaments)
% Splits a tree into a set of neurites by assigning labels to the Filament points.
%
% Input: Probs     - probability Image
%        Soma      - list of points of the image that are the soma
%                    in linear indexes
%        Centroid  - centroid of the soma
%        Filaments - list of points that form the tree
%
% Output: parents   - vector with the id of the parent point in the Filamet
%                     vector
%         neuriteId - index of the neurite the filament point belongs to.
%                     0 means it belongs to the soma
%         branchesLeafs - for each point it contains an int. Meaning
%                        -1 point inside the soma
%                         0 leaf
%                         1 regular point
%                         >1 bifurcation (number of sons)


% Finds the point that is closest to the centroid
[Y,X] = ind2sub(size(Probs), Filaments);
X= X - Centroid(1);
Y= Y - Centroid(2);
dist=(X.*X)+(Y.*Y);
[val, idxRoot] = min(dist);

PTImage = zeros(size(Probs));
PTImage(Filaments) = 1:length(Filaments);

weightMatrix = sparse(length(Filaments),length(Filaments));

edges=[];
[R,C] = ind2sub(size(Probs), Filaments);

for i = 1:length(Filaments)
    r = R(i);
    c = C(i);
    for ct = -1:1:1
        if (c-ct <= 0) || (c+ct > size(Probs,2))
            continue
        end
        for rt = -1:1:1
            if (r-rt <= 0) || (r+rt > size(Probs,1))
                continue
            end
            if(rt==0 && ct==0)
                continue;
            end
            idx =  PTImage(r+rt,c+ct);
            if idx ~= 0
                d = sqrt(rt*rt+ct*ct);
                w = (-log(Probs(Filaments(i))) - log(Probs(Filaments(idx))))/2 + 0.001;
                if(w <= 0)
                    str = sprintf('Weight of point %i =  %f as neighbor', idx,w);
                    disp(str);
                elseif (w>=5)
                    w=5;
                end
                weightMatrix(i,idx) = w*d;
            end
        end
    end
end

dA = mst_prim(weightMatrix,'full',idxRoot);


ImgSoma       = zeros(size(Probs));
ImgSoma(Soma) = 1;
parents       = zeros(size(Filaments));
neuriteId     = zeros(size(Filaments));
parents(idxRoot) = idxRoot;
[parents, neuriteId] = ...
    findParents(ImgSoma, Filaments, dA, idxRoot, parents, neuriteId);

branchesLeafs = zeros(size(Filaments));
for i = 1:1:length(parents)
    if( parents(i) > 0 && parents(i) < length(parents))
      branchesLeafs(parents(i)) = branchesLeafs(parents(i))+1;
    end
end

branchesLeafs(find(neuriteId==0)) = -1;





% Helper function
function [Parents, neuriteId] = ...
    findParents(ImgSoma, Vertex, AdjacencyMatrix, ptIdx, Parents, neuriteId)


neighbors = find(AdjacencyMatrix(ptIdx,:)~=0);

%% Finds the number of non-visited kids


for nn = 1:1:length(neighbors)
    numberKids = 0;

    % if the point has not been visited
    if Parents(neighbors(nn))== 0 
       numberKids = numberKids + 1;
       Parents(neighbors(nn)) = ptIdx; 

       % if the point is in the soma, it does not start a new id
       if ImgSoma( Vertex(neighbors(nn) ) ) == 1
          neuriteId( neighbors(nn) ) = 0;
       % if the parent was in the soma, it starts a new id
       elseif ImgSoma( Vertex(ptIdx) ) == 1
           neuriteId( neighbors(nn) ) = max(neuriteId)+1;
       % if not take the id of the parent
       else
           neuriteId( neighbors(nn) ) = neuriteId(ptIdx);
       end
       [Parents, neuriteId] = ...
           findParents...
            (ImgSoma, Vertex, AdjacencyMatrix, neighbors(nn), Parents, neuriteId);
    end
    

end




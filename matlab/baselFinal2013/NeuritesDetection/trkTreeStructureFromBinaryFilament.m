function [parents, numkids] = trkTreeStructureFromBinaryFilament(Filaments, Soma, imSize)

% [parents, numkids] = trkTreeStructureFromBinaryFilament(Filaments, Soma)
%
%
% Input: Filaments - list of points that form the tree
%                    the filament is assumed to be computed using the
%                    BackPropagate function that gearenties that:
%                       * todo
%                       * todo
%        Soma      - list of points of the image that are the soma
%                    in linear indexes
%
%
% Output: parents   - vector with the id of the parent point in the Filament
%                     vector
%         numkids   - for each point it contains an int. Meaning
%                        -1 point inside the soma
%                         0 leaf
%                         1 regular point
%                         >1 bifurcation (number of sons)

% first get the root point

rootPoints = intersect(Filaments, Soma);
if( isempty(rootPoints) )
    error('Soma and Filaments must intersect when Back propagation is used');
elseif(length(rootPoints) == 1)
    %perfect situation, only one root, continue
    rootIdx = find(Filaments==rootPoints);
else
    % they should be all connected(8-connectivity)
    % any of them should do
    rootIdx = find(Filaments==rootPoints(1));
end

% second get the adjacency matrix
PTImage = zeros(imSize);
PTImage(Filaments) = 1:length(Filaments);

weightMatrix = sparse(length(Filaments),length(Filaments));

[R,C] = ind2sub(imSize, Filaments);

for i = 1:length(Filaments)
    r = R(i);
    c = C(i);
    for ct = -1:1:1
        if (c-ct <= 0) || (c+ct > imSize(2))
            continue
        end
        for rt = -1:1:1
            if (r-rt <= 0) || (r+rt > imSize(1))
                continue
            end
            if(rt==0 && ct==0)
                continue;
            end
            idx =  PTImage(r+rt,c+ct);
            if idx ~= 0
                d = sqrt(rt*rt+ct*ct);
                w = 1;
                weightMatrix(i,idx) = w*d;%#ok
            end
        end
    end
end

dA = mst_prim(weightMatrix,'full',rootIdx);

parents       = zeros(size(Filaments));
parents(rootIdx) = -1;
[parents] = findDescendents(Filaments, dA, rootIdx, parents);

numkids = zeros(size(Filaments));
for i = 1:1:length(parents)
    if( parents(i) > 0 )
        numkids(parents(i)) = numkids(parents(i))+1;
    end
end

end



function [parents] = findDescendents(Filaments, dA, rootIdx, parents)

neighbors = find(dA(rootIdx,:)~=0);
for nn = 1:length(neighbors)
    if parents(neighbors(nn)) ==0
        parents(neighbors(nn)) = rootIdx;
        [parents] = findDescendents(Filaments, dA, neighbors(nn), parents);
    end
end

end


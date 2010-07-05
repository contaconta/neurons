function FEAT_CANONICAL = ray3D(E,r,c,z, V)

% temp
E = double(E > 0);

icosahedron = [ -0.577350269189626,  -0.577350269189626,  -0.577350269189626;...
                -0.577350269189626,  -0.577350269189626,   0.577350269189626;...
                -0.577350269189626,   0.577350269189626,  -0.577350269189626;...
                -0.577350269189626,   0.577350269189626,   0.577350269189626;...
                0.577350269189626,  -0.577350269189626,  -0.577350269189626;...
                0.577350269189626,  -0.577350269189626,   0.577350269189626;...
                0.577350269189626,   0.577350269189626,  -0.577350269189626;...
                0.577350269189626,   0.577350269189626,   0.577350269189626;...
                0,  -0.356822089773090,  -0.934172358962716;...
                0,  -0.356822089773090,   0.934172358962716;...
                0,   0.356822089773090,  -0.934172358962716;...
                0,   0.356822089773090,   0.934172358962716;...
                -0.356822089773090,  -0.934172358962716, 0;...
                -0.356822089773090,   0.934172358962716, 0;...
                0.356822089773090,  -0.934172358962716,  0;...
                0.356822089773090,   0.934172358962716,  0;...
                -0.934172358962716,  0,  -0.356822089773090;...
                -0.934172358962716,  0,   0.356822089773090;...
                0.934172358962716,   0,  -0.356822089773090;...
                0.934172358962716,   0,   0.356822089773090];
faces = size(icosahedron,1);

% icosahedron = [1 0 0; 
%                -1 0 0;
%                0 1 0;
%                0 -1 0;
%                0 0 1;
%                0 0 -1];
% faces = size(icosahedron,1);

            
temp = icosahedron;
icosahedron(:,1) = temp(:,2);
icosahedron(:,2) = temp(:,1);
            
FEAT = zeros(faces, 1);
            
endpoints = zeros(size(icosahedron));

Edraw = E;

for i = 1:faces
    
    [R C Z] = bresenham_line(icosahedron(i,:), r, c, z, E);
    
    FEAT(i) = length(R);
    
    ico_ind = i - 1;
    
    %if ico_ind == 0
    %disp([ 'icosahedron ' num2str(ico_ind)]);
    
    endpoints(i,:) = [R(length(R)) C(length(C)) Z(length(Z))];
    %disp(['   endpoint = ' num2str(endpoints(i,:))]);
    
    
    % draw rays on E
    for j = 1:length(R)
        if (R(j) > 0) && (R(j) <= size(E,1)) && (C(j) > 0) && (C(j) <= size(E,2)) && (Z(j) > 0) && (Z(j) <= size(E,3))
            if i ~= 1
                Edraw(R(j),C(j),Z(j)) = 7; 
            else
                %E(R(j),C(j),Z(j)) = 10; 
                Edraw(R(j),C(j),Z(j)) = 7; 
            end
            if j == length(R)
                Edraw(R(j),C(j),Z(j)) = 12;
            end
        end
    end
end

figure; hold on;


% zeromean = mean(endpoints,1);
% ptcloud = endpoints - repmat(zeromean,[faces 1]);
cv = cov(endpoints);
%cv = (1/(faces -1 )) * (ptcloud'*ptcloud);
cv

% compute the covariance
cv = cov(endpoints);

% eigenvalue decomposition
[V,D] = eigs(cv);

a = (pi/3) -.4;
V = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)]';

% rotation vector from eigen values
R = V';

[D, V]

% rotated icosahedron
icorot = R*icosahedron';
icorot = icorot';



            
wicorot(:,1) = FEAT.*icorot(:,1); 
wicorot(:,2) = FEAT.*icorot(:,2);
wicorot(:,3) = FEAT.*icorot(:,3);

% eigenvectors
%quiver3(repmat(c-.5,[3 1]), repmat(r-.5,[3 1]), repmat(z-.5,[3 1]), 25*V(:,1), 25*V(:,2), 25*V(:,3) , 'LineWidth', 2, 'Color', [0 1 0]);

% rotated icosahedron
%quiver3(repmat(c-.5,[faces,1]), repmat(r-.5,[faces 1]), repmat(z-.5,[faces 1]), wicorot(:,1),wicorot(:,2), wicorot(:,3) );

% first icosahedron element is highlighted
%h = quiver3(repmat(c-.5,[1,1]), repmat(r-.5,[1 1]), repmat(z-.5,[1 1]), wicorot(1,1), wicorot(1,2), wicorot(1,3) , 'LineWidth', 3, 'Color', [1 0 0]);

xlabel('x');
ylabel('y');
zlabel('z');


new_inds = zeros(size(FEAT));
new_inds2 = zeros(size(FEAT));

% search for matches
for i = 1:faces
    d = zeros(1,faces);
    d2 = zeros(1,faces);
    
    for j = 1:faces    
        d2(j) = acos((icorot(i,:) * icosahedron(j,:)') / (norm(icorot(i,:)) * norm(icosahedron(j,:))));    
    end
    
    for j = 1:faces    
        d(j) = sum((icorot(i,:) - icosahedron(j,:)).^2);
    end
    
    [v min_ind] = min(abs(d));
    [v2 min_ind2] = min(abs(d2));
    
    if length(find(d == v)) > 1
        disp(' two or more icosahedron matches were found. fix this case!');
        keyboard;
    end
    
    new_inds(i) = min_ind;
    new_inds2(i) = min_ind2;
end


FEAT_CANONICAL = FEAT(new_inds);


new_inds(1)
    
[R C Z] = bresenham_line(icosahedron(new_inds(1),:), r, c, z, E);
% highlight the first icosahedron point
for j = 1:length(R)
    if (R(j) > 0) && (R(j) <= size(E,1)) && (C(j) > 0) && (C(j) <= size(E,2)) && (Z(j) > 0) && (Z(j) <= size(E,3))
        
        Edraw(R(j),C(j),Z(j)) = 10; 
       
    end
end
    
    
    
%keyboard;

vol3d('cdata', Edraw, 'texture', '3D'); grid on;



% quiver3(repmat(c,[faces,1]), repmat(r,[faces 1]), repmat(z,[faces 1]), 50*icorot(:,1), 50*icorot(:,2), 50*icorot(:,3) );
% h = quiver3(repmat(c,[1,1]), repmat(r,[1 1]), repmat(z,[1 1]), 50*icorot(1,1), 50*icorot(1,2), 50*icorot(1,3) , 'LineWidth', 3, 'Color', [1 0 0]);

keyboard;            










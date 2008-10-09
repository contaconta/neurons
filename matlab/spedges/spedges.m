function A = spedges(I, angles, sigma)

A.angle = angles;

for i = 1:length(angles)
    [A.spedges(i,:,:), A.edge] = spedge_dist(I, angles(i),sigma, 'count');   
end


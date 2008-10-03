function A = spedges(I, angles, sigma)

A.angle = angles;

for i = 1:length(angles)
    A.spedges(i,:,:) = spedge_dist(I, angles(i) ,sigma);
end


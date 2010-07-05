figure; quiver3(zeros(20,1),zeros(20,1),zeros(20,1),icosahedron(:,1), icosahedron(:,2), icosahedron(:,3), 'r.');
hold on;
quiver3(zeros(20,1),zeros(20,1),zeros(20,1),icorot(:,1), icorot(:,2), icorot(:,3), 'b.');


for i = 1:length(new_inds)
    %plot3([icorot(i,1); icosahedron(new_inds(i),1)], [icorot(i,2); icosahedron(new_inds(i),2)], [icorot(i,3); icosahedron(new_inds(i),3)], 'g.-');
    
    pt1 = [icorot(i,1) icorot(i,2) icorot(i,3)];
    pt2 = [icosahedron(new_inds(i),1) icosahedron(new_inds(i),2) icosahedron(new_inds(i),3)];
    vec = pt2 - pt1;
    
    h = quiver3(pt1(1), pt1(2), pt1(3), vec(1), vec(2), vec(3),  'g');
    c = get(h, 'Children');
    set(c(2), 'LineWidth', 5);
    text(pt1(1), pt1(2), pt1(3), num2str(norm(vec)));
end

axis equal
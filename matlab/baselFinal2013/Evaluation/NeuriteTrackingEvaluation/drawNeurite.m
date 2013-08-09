function [line_r, line_c] = drawNeurite(Parents,NeuritePixelIdxList,NumKids)

% convert from inds to coordinates
C           = 696;
R           = 520;
[rlist, clist] = ind2sub([R C], NeuritePixelIdxList);

% find the root point
root_ind = find(Parents == -1);
root_c = clist(root_ind);
root_r = rlist(root_ind);

% find the extreme point from the endpoints
endpoint_inds = find(NumKids == 0);
for i = 1:numel(endpoint_inds)
    leaf_c = clist(endpoint_inds(i));
    leaf_r = rlist(endpoint_inds(i));
    dist_to_root(i) = sqrt( (leaf_c -root_c)^2 + (leaf_r - root_r)^2);
end
[val, max_ind] = max(dist_to_root); %#ok<ASGLU>
extreme_ind = endpoint_inds(max_ind); 
extreme_c = clist(extreme_ind);
extreme_r = rlist(extreme_ind);


% follow the tree to get the line from extreme to root
ind = extreme_ind;
line_c = extreme_c;
line_r = extreme_r;
i = 1;
parent_ind = Parents(ind);
% fprintf('%d  r=%d  c=%d\n', ind, line_r(i), line_c(i));
while parent_ind > 0
    i = i + 1;
    ind = Parents(ind);
    line_c(i) = clist(ind);
    line_r(i) = rlist(ind);
    parent_ind = Parents(ind);
%     fprintf('%d  r=%d  c=%d\n', ind, line_r(i), line_c(i));
end


% % visualization
% B = zeros(R,C);
% B(NeuritePixelIdxList) = 3;
% for i = 1:numel(endpoint_inds)
%     B(NeuritePixelIdxList(endpoint_inds(i))) = 2;
% end
% B(NeuritePixelIdxList(root_ind)) = 1;
% figure; imagesc(B); 
% hold on;
% line(line_c(:), line_r(:), 'Color', [1 1 1], 'LineWidth', 2);




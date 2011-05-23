function DIST = getDistanceTraveled(D, id, T)

tlist = zeros(numel(T),1);
DIST = tlist;

for d = 1:length(D)

   if D(d).ID == id 
       t = D(d).t;
       tlist(t) = d;
   end
end


firstAnn = find(tlist,1);

if ~isempty(firstAnn)
    if length(tlist) > 1
        dlist = tlist(find(tlist));
    
        for d = 2:length(dlist)
            
            
            loc2 = [D(dlist(d)).Centroid D(dlist(d)).zCenter];
            loc2 = loc2(:);
            t2 = D(dlist(d)).t;
            loc1 = [D(dlist(d-1)).Centroid D(dlist(d)).zCenter];
            loc1 = loc1(:);
            t1 = D(dlist(d-1)).t;
            
            euclid_dist = dist([loc2,loc1]);
            DIST(t2) = euclid_dist(2,1) / abs(t2-t1);
            %keyboard;
        end
    end
end
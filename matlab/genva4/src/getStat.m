function stat = getStat(D, id, statname, T)


stat = zeros(numel(T),1);

for d = 1:length(D)

   if D(d).ID == id 
        t = D(d).t;
        stat(t) = D(d).(statname);       
   end
    
end
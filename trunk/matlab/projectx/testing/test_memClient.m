
width = 2000
height = 1000

% Start the memory daemon
system('killall memDaemon');
system(['./bin/memDaemon ' int2str(width) ' ' int2str(height) ' int &']);

clear A
A=zeros(height,width);
k = 0;
for i=1:height
  for j=1:width
    A(i,j) = k;
    k = k + 1;
  end
end
A=uint32(A);

for i=1:size(A,1)
  mexStoreResponse(A(i,:),'row',i,'HA');
end

B=zeros(size(A));
for i=1:size(A,1)
  B(i,:)=mexLoadResponse('row',i,'HA');
end

if(A==B)
  display 'Stored and retrieved values are identical'
else
  display 'Stored and retrieved values are different'
end

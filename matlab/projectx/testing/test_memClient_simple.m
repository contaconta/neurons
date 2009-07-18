width = 200
height = 100

% Start the memory daemon
% system('killall memDaemon');
% system(['./bin/memDaemon ' int2str(width) ' ' int2str(height) ' int &']);

clear A
A=zeros(height,width);
B = A;
k = 0;
for i=1:height
  for j=1:width
    A(i,j) = k;
    k = k + 1;
  end
end
A=uint32(A);

i=1
mexStoreResponse(A(i,:),'row',i,'HA');
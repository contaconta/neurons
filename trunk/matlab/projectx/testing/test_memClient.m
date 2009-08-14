% 
% width = 200
% height = 100
% 
% % Start the memory daemon
% system('killall memDaemon');
% system(['./bin/memDaemon ' int2str(width) ' ' int2str(height) ' int &']);
% 
% clear A
% A=zeros(height,width, 'uint32');
% B = A;
% k = 0;
% for i=1:height
%   for j=1:width
%     A(i,j) = uint32(k);
%     k = k + 1;
%   end
% end
% %A=uint32(A);
% 
% for i=1:size(A,1)
%   mexStoreResponse(A(i,:),'row',i,'HA');
% end
% 
% 
% for i=1:size(A,1)
%   B(i,:)=mexLoadResponse('row',i,'HA');
% end
% 
% 
% if(A==B)
%   display 'Stored and retrieved values are identical'
% else
%   display 'Stored and retrieved values are different'
% end
% 

width = 19470;
height = 5598;

%Start the memory daemon
system('killall memDaemon');
system(['./bin/memDaemon ' int2str(width) ' ' int2str(height) ' int &']);

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
A=int32(A);

for i=1:size(A,1)
  mexStoreResponse(A(i,:),'row',i,'HA');
end
for i=1:size(A,1)
  mexStoreResponse(A(i,:),'row',i,'HA');
end

tic;

for i=1:90000
    rowsamp = ceil( height * rand(1));
    ROW = mexLoadResponse('row',rowsamp,'HA')';
    %disp([ 'i = ' num2str(i) ]);
    if ROW ~= A(rowsamp,:)
        disp('rows are not equal!');
        keyboard;
    end
    %pause(0.001);
end

disp('it worked!'); toc;

% for i=1:size(A,1)
%   B(i,:)=mexLoadResponse('row',i,'HA');
% end
% 
% 
% if(A==B)
%   display 'Stored and retrieved values are identical'
% else
%   display 'Stored and retrieved values are different'
% end

% points = load('neuronTrain/points.txt');
% points = load('points.txt');
% als = load('alpha_coords.txt');

data = loadTorchFile('../neurons/n1/training.txt');

pos = data(find(data(:,size(data,2)) == 1),1:size(data,2)-1);
neg = data(find(data(:,size(data,2)) ==-1),1:size(data,2)-1);

m_p = mean(pos);
m_n = mean(neg);

Sw = zeros(size(pos,2));

for i=1:1:size(pos,1)
    Sw = Sw + (pos(i,:)-m_p)'*(pos(i,:)-m_p);
end

for i=1:1:size(neg,1)
    Sw = Sw + (neg(i,:)-m_n)'*(neg(i,:)-m_n);
end

f = Sw^-1*(m_p-m_n)';

n = sqrt(f'*f);
f = f./n;

% figure;
% hold on;
% for nD = 1:1:size(pos,2)
%     plot(pos(:,nD), nD, 'b+','MarkerSize', 3);
%     plot(neg(:,nD), nD, 'r*','MarkerSize', 3);
% end
% plot(pos*f,0,'b+');
% plot(neg*f,0,'r*');


save('../neurons/n1/fisher.txt','f','-ascii');

% quit

% fshr = [als*f , points(:,5)];

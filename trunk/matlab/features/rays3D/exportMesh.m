addpath('polygon2voxel_version1f');

load sphere
figure, patch(FV,'FaceColor',[1 0 0]); axis square;
Volume=polygon2voxel(FV,[50 50 50],'auto');

Volume = Volume*255;

fid = fopen('sphere.dat','w');
%fwrite(fid,Volume,'uint8');
for i=1:50
  fwrite(fid,Volume(:,:,i),'uint8');
  img = imresize(Volume(:,:,i),2);
  imwrite(img,['sphere' sprintf('%.2d',i) '.png']);
end
fclose(fid);

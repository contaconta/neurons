img_name = {'mitochondria2.png'};

%for angle = 0:2:180
for angle = 85:1:95
  
  angle
  
  rays(img_name,20,angle);

  %g = imread('g.png');
  %imagesc(g);

  im = imread('ray1.png');
  imagesc(im);
  refresh
  pause(0.5)
end

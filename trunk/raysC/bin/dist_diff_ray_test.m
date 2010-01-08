
img_name = {'mitochondria1.png'};
sigma = 20;
start_angle = 0;
end_angle = 180;
step_angle = 90;
edge_low_threshold = 10000;
edge_high_threshold = 30000;
distDiffRays(img_name,sigma,start_angle,end_angle,step_angle,edge_low_threshold,edge_high_threshold);

g = imread(img_name{1});
%for i=start_angle:step_angle:end_angle
%    im=read_32bitsimage(['ray1_' num2str(i) '.ppm'],[size(g,2) size(g,1)]);
%    imagesc(im);
%    refresh;
%    pause(0.1);
%end

angles = start_angle:step_angle:end_angle;
c = combnk(angles,2);

for i=1:size(c,1)
    a = c(i,1);
    b = c(i,2);
    subplot(3,1,1);
    im=read_32bitsimage(['ray1_' num2str(a) '.ppm'],[size(g,2) size(g,1)]);
    imagesc(im);
    subplot(3,1,2);
    im=read_32bitsimage(['ray1_' num2str(b) '.ppm'],[size(g,2) size(g,1)]);
    imagesc(im);
    subplot(3,1,3);
    im=read_floatimage(['ray2_' num2str(a) '_' num2str(b) '.ppm'],[size(g,2) size(g,1)]);
    imagesc(im);
    refresh;
    pause(0.1);
end

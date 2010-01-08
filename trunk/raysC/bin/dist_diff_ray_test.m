%function ray_test(angle)

img_name = {'mitochondria1.png'};
sigma = 20;
start_angle = 0;
end_angle = 30;
step_angle = 2;
edge_low_threshold = 13000;
edge_high_threshold = 35000;
distDiffRays(img_name,sigma,start_angle,end_angle,step_angle,edge_low_threshold,edge_high_threshold);

g = imread(img_name{1});
for i=2:2:30
    im=read_32bitsimage(['ray1_' num2str(i) '.ppm'],[size(g,2) size(g,1)]);
    imagesc(im);
    refresh;
    pause(0.1);
end

a = 0;
b = 2;
    im=read_floatimage(['ray2_' num2str(a) '_' num2str(b) '.ppm'],[size(g,2) size(g,1)]);
    imagesc(im);
    refresh;
    pause(0.1);


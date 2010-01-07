%function ray_test(angle)

img_name = {'mitochondria1.png'};
sigma = 20;
start_angle = 0;
end_angle = 30;
step_angle = 2;
edge_low_threshold = 13000;
edge_high_threshold = 35000;
distDiffRays(img_name,sigma,start_angle,end_angle,step_angle,edge_low_threshold,edge_high_threshold);


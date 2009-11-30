function getFinalOutput(image_name, mask_name)

im=imread(['~/share/Data/LabelMe/Images/FIBSLICE/' image_name '.png']);
mask=imread(mask_name);
mask=mask(:,:,1);
base_name = mask_name(1:length(mask_name)-4);
disp(['Creating image ' base_name 'overlay.png'])
%M = postprocessing(mask,im,1);
%print('-dpng','-r300', [base_name '-overlay.png'])
%M = postprocessing(mask,im,0.6);
%print('-dpng','-r300', [base_name '-overlay60.png'])
%M = postprocessing(mask,im,0.7);
%print('-dpng','-r300', [base_name '-overlay70.png'])
M = postprocessing(mask,im,0.8);
print('-dpng','-r300', [base_name '-overlay80.png'])


quit

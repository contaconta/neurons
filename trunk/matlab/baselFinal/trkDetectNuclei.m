function Nuclei = trkDetectNuclei(R, SIGMA_RED, minArea, maxArea, MSER_MaxVariation, MSER_Delta)


TMAX = length(R);
Nuclei = cell(size(R));
EightBitsImages = cell(size(R));

parfor  t = 1:TMAX
    Rt = mat2gray(double(R{t}));
    I = imgaussian(Rt, SIGMA_RED);
	I = uint8(255*mat2gray(I));
    EightBitsImages{t} = I;
    Nuclei{t} = vl_mser_new4(I, 'MinDiversity', minArea/maxArea,...
                                'MaxVariation', MSER_MaxVariation,...
                                'MinArea', minArea/numel(I), ...
                                'MaxArea', maxArea/numel(I), ...
                                'BrightOnDark', 1, ...
                                'Delta',MSER_Delta) ;
end
%%
for t = 1:TMAX
    mm = zeros(size(R{1}));
    for x = Nuclei{t}'
        s = vl_erfill(EightBitsImages{t}, x);
        mm(s) = mm(s)+1;
    end
    Nuclei{t} = mm;
    Nuclei{t}  	= imfill(Nuclei{t} > 0, 'holes');
    Nuclei{t} = bwlabel(Nuclei{t});
end
clear EightBitsImages;
function G = gaussianKernel(Image, center,r, Kernel)

G = zeros(size(Image));

if center(1)-2*r >= 1
    startXG = center(1)-2*r;
    startXK = 1;
else
    startXG = 1;
    startXK = 2+2*r-center(1);
end

if center(2)-2*r >= 1
    startYG = center(2)-2*r;
    startYK = 1;
else
    startYG = 1;
    startYK = 2+2*r-center(2);
end

if center(1)+2*r <= size(Image, 1)
    endXG = center(1)+2*r;
    endXK = size(Kernel, 1);
else
    endXG = size(Image, 1);
    endXK = size(Image, 1) +1 +2*r-center(1);
end

if center(2)+2*r <= size(Image, 2)
    endYG = center(2)+2*r;
    endYK = size(Kernel, 2);
else
    endYG = size(Image, 2);
    endYK = size(Image, 2) +1 +2*r-center(2);
end

G(startXG:endXG, startYG:endYG) = Kernel(startXK:endXK, startYK:endYK);


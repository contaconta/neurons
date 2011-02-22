function [x y w s] = getMyGaussiansFromFeth(Mixture, PAD, IMSIZE)

 % put the GMM into a format I like
    x = Mixture.Mu(2,:) - PAD;
    x = x(:);
    y = Mixture.Mu(1,:) - PAD;
    y = y(:);
    w = Mixture.Weights;
    s = Mixture.Sigmas;

    % remove any gaussians placed outside of the image due to padding
    badinds1 = (x < 1);
    badinds2 = (y < 1);
    badinds3 = (x > IMSIZE(2));
    badinds4 = (y > IMSIZE(1));
    inds = ~(badinds1 | badinds2 | badinds3 | badinds4);
    x = x(inds);
    y = y(inds);
    w = w(inds);
    s = s(inds);

    % adjust for 0-indexing
    x = x - 1;
    y = y - 1;
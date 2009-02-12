function gau = gauss0(sigma)


% Magic numbers
  GaussianDieOff = .0001;  
  %PercentOfPixelsNotEdges = .7; % Used for selecting thresholds
  %ThresholdRatio = .4;          % Low thresh is this fraction of the high.
  
  % Design the filters - a gaussian and its derivative
  
  pw = 1:30; % possible widths
  ssq = sigma^2;
  width = find(exp(-(pw.*pw)/(2*ssq))>GaussianDieOff,1,'last');
  if isempty(width)
    width = 1;  % the user entered a really small sigma
  end

  t = (-width:width);
  gau = exp(-(t.*t)/(2*ssq))/(2*pi*ssq);     % the gaussian 1D filter


% 
% 
% %G = gauus0(sigma)
% %
% %Gives a 1-D approximate gaussian filter, G, with an 
% %appropriately sized window length for its standard deviation, sigma.
% 
% g0 = normpdf(-100:100, 0, sigma);
% 
% g0 = g0(find(g0 > .0000001 * max(g0)))';
% 
% 


%%% VERY VERY IMPORTANT GAUSS AND DERIVATIVE %%%%
% g=normpdf([-100:100],0,10);
% plot(g)
% hold on
% g=normpdf([-100:100],0,20);
% plot(g,'r')
% 
% % first derivative of Gaussian
% clf
% len=length(g)
% dg=g(2:len)-g(1:len-1);
% plot(dg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

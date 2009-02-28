function EDGE = edgemethods(I, ind)
%
%
%
%
%


switch ind
    
    %% CANNY, SIGMA = .3, thresh = .1 - .9
    case 1
        sigma = .3;
        thresh = .1;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 2
        sigma = .3;
        thresh = .3;
        EDGE = edge(I,'canny', thresh, sigma);
    case 3
        sigma = .3;
        thresh = .5;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 4
        sigma = .3;
        thresh = .7;
        EDGE = edge(I,'canny', thresh ,sigma);
	case 5
        sigma = .3;
        thresh = .9;
        EDGE = edge(I,'canny', thresh ,sigma);
        
    %% CANNY, SIGMA = .5, thresh = .1 - .9
    case 6
        sigma = .5;
        thresh = .1;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 7
        sigma = .5;
        thresh = .3;
        EDGE = edge(I,'canny', thresh, sigma);
    case 8
        sigma = .5;
        thresh = .5;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 9
        sigma = .5;
        thresh = .7;
        EDGE = edge(I,'canny', thresh ,sigma);
	case 10
        sigma = .5;
        thresh = .9;
        EDGE = edge(I,'canny', thresh ,sigma);
        
    %% CANNY, SIGMA = .8, thresh = .1 - .9
   	case 11
        sigma = .8;
        thresh = .1;
        EDGE = edge(I,'canny', thresh ,sigma);
   	case 12
        sigma = .8;
        thresh = .3;
        EDGE = edge(I,'canny', thresh ,sigma);
   	case 13
        sigma = .8;
        thresh = .5;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 14
        sigma = .8;
        thresh = .7;
        EDGE = edge(I,'canny', thresh ,sigma);
	case 15
        sigma = .8;
        thresh = .9;
        EDGE = edge(I,'canny', thresh ,sigma);
        
    %% LOG, zero-crossings
    case 16
        sigma = 1;
        EDGE = edge(I, 'log', 0, sigma); 
    case 17
        sigma = 1.25;
        EDGE = edge(I, 'log', 0, sigma); 
   	case 18
        sigma = 1.5;
        EDGE = edge(I, 'log', 0, sigma); 
  	case 19
        sigma = 1.75;
        EDGE = edge(I, 'log', 0, sigma); 
    case 20
        sigma = 2;
        EDGE = edge(I, 'log', 0, sigma); 
    case 21
        sigma = 2.5;
        EDGE = edge(I, 'log', 0, sigma); 
    case 22
        sigma = 3;
        EDGE = edge(I, 'log', 0, sigma); 
        
    %% SOBEL, thresholds 0.01 - 0.1
    case 23
        thresh = .01;
        EDGE = edge(I,'sobel', thresh);
    case 24
        thresh = .02;
        EDGE = edge(I,'sobel', thresh);
   	case 25
        thresh = .04;
        EDGE = edge(I,'sobel', thresh);
   	case 26
        thresh = .06;
        EDGE = edge(I,'sobel', thresh);
   	case 27
        thresh = .08;
        EDGE = edge(I,'sobel', thresh);
    case 28
        thresh = .1;
        EDGE = edge(I,'sobel', thresh);
        
        
    %% MORE SOBEL THRESHOLDS
    case 29
        thresh = .001;
        EDGE = edge(I,'sobel', thresh);
    case 30
        thresh = .005;
        EDGE = edge(I,'sobel', thresh);
    case 31
        thresh = .0075; 
        EDGE = edge(I,'sobel', thresh);
    case 32
        thresh = .02;
        EDGE = edge(I,'sobel', thresh);

        
    %% MORE CANNY THRESHOLDS
    case 33 
        sigma = .8;
        thresh = .01;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 34
        sigma = .8;
        thresh = .025;
        EDGE = edge(I,'canny', thresh ,sigma);
    case 35
        sigma = .8;
        thresh = .05;
        EDGE = edge(I,'canny', thresh ,sigma);
                
    %% JUST THRESHOLD AND INVERT THE IMAGE
    case 36
        EDGE = (I <.5);

end

EDGE = bwmorph(EDGE, 'diag');
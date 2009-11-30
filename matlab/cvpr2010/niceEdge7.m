
function E = niceEdge7(I, varargin)

[E O] = canny(I, 6); 
E = nonmaxsup(E,O,3) > 2;


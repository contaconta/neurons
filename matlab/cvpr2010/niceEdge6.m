
function E = niceEdge6(I, varargin)

[E O] = canny(I, 4); 
E = nonmaxsup(E,O,3) > 4;
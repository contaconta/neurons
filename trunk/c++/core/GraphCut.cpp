#include "GraphCut.h"

ostream& operator <<(ostream &os,const Point3Di &point)
{
    for(int i=0;i<point.coords.size();i++)
    {
        if(i==point.coords.size()-1)
            os << point.coords[i]<<endl;
        else
            os << point.coords[i] << " ";
    }
    return os;
}

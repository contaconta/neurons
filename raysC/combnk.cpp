//#include <iostream>
#include <string>
#include "combnk.h"

void add(vii begin,vii end,vector< vector<int> >& r)
{
  vector<int> temp;
  for (vii it=begin;it!=end;++it)
    temp.push_back(*it);

  r.push_back(temp);
}

/* Recursive template function
 * The typical way of filling out the parameters in n_column and r_column is always 0
 * @param loop is the number of elements in r sequence minus that of n sequence
 */
void recursive_combination(vii nbegin, vii nend, int n_column,
                           vii rbegin, vii rend, int r_column,
                           int loop, vector< vector<int> >& res)
{
  int r_size=rend-rbegin;

  int localloop=loop;
  int local_n_column=n_column;

  //A different combination is out
  if(r_column>(r_size-1))
  {
    add(rbegin,rend,res);
    return;
  }
  //===========================

  for(int i=0;i<=loop;++i)
  {
    vii it1=rbegin;
    for(int cnt=0;cnt<r_column;++cnt)
    {
      ++it1;
    }
    vii it2=nbegin;
    for(int cnt2=0;cnt2<n_column+i;++cnt2)
    {
      ++it2;
    }

    *it1=*it2;

    ++local_n_column;

    recursive_combination(nbegin,nend,local_n_column,
                          rbegin,rend,r_column+1,localloop,res);

    --localloop;
  }
}

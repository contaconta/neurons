
#include <vector>

using namespace std;

typedef vector<int>::iterator vii;

void recursive_combination(vii nbegin, vii nend, int n_column,
                           vii rbegin, vii rend, int r_column,int loop, vector< vector<int> >& res);

#include <stdio.h>
#include <vector>

using namespace std;

//T in float, int
template< class T>
class Cont
{
public:
  T x;

  Cont(T _x){
    x = _x;
  }

  void print(int n){
    printf("Int: %i\n", n);
  }

  void print(float f){
    printf("Float: %f\n",f);
  }

  void print(){
    print(x);
  }
};


int main(int argc, char **argv) {

  Cont<int>* ci = new Cont<int>(3);
  Cont<float>* cf = new Cont<float>(3.5);

  ci->print();
  cf->print();
}

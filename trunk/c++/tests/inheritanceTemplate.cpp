#include <stdio.h>
#include <vector>

template <class T=float>
class Base
{
public:

  float w;

  Base(){
    w = 0;
  }

  Base(T _w){
    w = _w;
  }
};


template <class T>
class Deriv : public Base< T >
{
public:

  Deriv() : Base<T>(){};

  Deriv(T _w) : Base<T>(_w){};

};


int main(int argc, char **argv) {

  Deriv< float >* der = new Deriv<float>(2.5);

  printf("%f\n", der->w);
}

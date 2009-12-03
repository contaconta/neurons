#include <stdio.h>
#include <vector>

using namespace std;

class Beep
{
public:
  void beep(){
    printf("Beep\n");
  }
};

class Up
{
public:
  // virtual void print(){
    // printf("Up\n");
  // }
  virtual void print() {printf("Up\n");}

  virtual void printUp(){
    print();
  }

};

class A : public Up, public Beep
{
public:
  int x;

  A(){x=3;}

  void print(){
    printf("A: x=%i\n",x);
  }
};

class B : public Up, public Beep
{
public:
  int x;

  B(){x=3;}

  void print(){
    printf("B: x=%i\n",x);
  }
};

class C : public A
{
public:
  int x;

  C(){x=3;}

  void print(){
    printf("C: x=%i\n",x);
  }
};



void printv(vector< Up* > v){
  for(int i = 0; i < v.size(); i++)
    v[i]->printUp();
}


int main(int argc, char **argv) {

  // Up* up = new Up();
  A*  a = new A();
  B*  b = new B();
  A*  c = new C();
  Up* u = new Up();

  vector< Up* > v;
  // v.push_back(up);
  v.push_back(a);
  v.push_back(b);
  v.push_back(c);
  v.push_back(u);
  printv(v);

  printf("And Up says ");
  u->print();
  printf("\n");

}

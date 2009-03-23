/** Class Object
 *  Simple abstract class that defines an object that can save itself.
 *
 *  German Gonzalez
 *  20080624
 */

#ifndef OBJECT_H_
#define OBJECT_H_

#include <fstream>
#include <string>
#include <assert.h>

using namespace std;

class Object
{
public:
  virtual void save(ostream &out)=0;

  virtual bool load(istream &in) =0;

  virtual void saveToFile(string filename){
    std::ofstream out(filename.c_str());
    save(out);
    out.close();
  }

  virtual void loadFromFile(string filename){
    std::ifstream in(filename.c_str());
    load(in);
    in.close();
  }

  virtual string className(){
    return "Object";
  }
};

#endif

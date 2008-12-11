#include "utils.h"

bool fileExists(string filename){
  // FILE* f = fopen(filename.c_str(), "r");
  // if(f!=NULL){
    // fclose(f);
    // return true;
  // }
  // return false;
  ifstream inp;
  inp.open(filename.c_str(), ifstream::in);
  if(inp.fail()){
    inp.close();
    return false;
  }
  inp.close();
  return true;
}

string getDirectoryFromPath(string path){
  size_t pos = path.find_last_of("/\\");
  if(pos == string::npos)
    return "./";
  else
    return path.substr(0,pos+1);
}

string getNameFromPath(string path){
  return path.substr(path.find_last_of("/\\")+1);
}

string getNameFromPathWithoutExtension(string path){
  string nameWith =  path.substr(path.find_last_of("/\\")+1);
  string nameWithout = nameWith.substr(0,nameWith.find_last_of("."));
  return nameWithout;
}


string getExtension(string path){
  return path.substr(path.find_last_of(".")+1);
}


string getDerivativeName(int order_x, int order_y, int order_z,
                         float sigma_x, float sigma_y, float sigma_z,
                         string directory)
{
  return "";
}

int   getFileSize(string path)
{
  ifstream is;
  is.open(path.c_str(), ios::binary);
  if(is.fail()){
    is.close();
    printf("Error getting the size of %s\n",path.c_str());
    return false;
  }
  is.seekg(0, ios::end);
  int size = is.tellg();
  is.close();
  return size;
}


vector< vector< double > > loadMatrix(string filename)
{
  assert(fileExists(filename));
  std::ifstream in(filename.c_str());
  if(!in.good())
    {
      printf("loadMatrix::The file %s can not be opened\n",filename.c_str());
      exit(0);
    }
  vector< vector< double > > toReturn = loadMatrix(in);
  in.close();
  return toReturn;
}

vector< vector< double > > loadMatrix(istream &file){
  vector< vector< double > > toReturn;
  int pos;
  string s;
  int matrix_width = 0;
  while(getline(file,s))
    {
      vector< double > vd;
      stringstream ss(s);
      double d;
      while(!ss.fail()){
        ss >> d;
        if(!ss.fail()){
          vd.push_back(d);
        }
      }
      if(matrix_width==0)
        matrix_width = vd.size();
      if(vd.size() == matrix_width){
        toReturn.push_back(vd);
      }else{
        // Return before the string that has been read
        file.seekg(pos);
        break;
      }
      pos = file.tellg();
    }
  return toReturn;
}


int factorial_n(int n){
  int ret = 1;
  for(int i = 1; i <= n; i++)
    ret = ret*i;
  return ret;
}

double combinatorial(int over, int under)
{
  int num = 1;
  for(int i = over; i > under; i--)
    num = num*i;
  int den = 1;
  for(int i = over - under; i > 1; i--)
    den = den*i;
  return double(num)/den;
}

int dfactorial_n(int n){

  if(n < -1){
    printf("dfactorial not implemented for negative numbers else than -1\n");
    return 0;
  }
  if( n == -1)
    return 1;
  int ret = 1;
  while( n > 0){
    ret *= n;
    n = n-2;
  }
  return ret;
}


bool isNumber(string s){
  std::istringstream inpStream(s);
  double inValue = 0.0;
  if(inpStream >> inValue)
    return true;
  else
    return false;
}


void saveVectorDouble(vector< double > &vc, string filename){
  std::ofstream out(filename.c_str());
  for(int i = 0; i < vc.size(); i++)
    out << vc[i] << std::endl;
  out.close();
}


void saveFloatVector( vector< float >& vc, string filename){
  std::ofstream out(filename.c_str());
  for(int i = 0; i < vc.size(); i++)
    out << vc[i] << std::endl;
  out.close();
}

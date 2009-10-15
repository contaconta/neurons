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

int get_files_in_dir(string dir, vector<string> &files)
{
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL) {
    files.push_back(string(dirp->d_name));
  }
  closedir(dp);
  return 0;
}

string getDerivativeName(int order_x, int order_y, int order_z,
                         float sigma_x, float sigma_y, float sigma_z,
                         string directory)
{
  string ret = directory + "/g";
  for(int i = 0; i < order_x; i++)
    ret = ret + "x";
  for(int i = 0; i < order_y; i++)
    ret = ret + "y";
  for(int i = 0; i < order_z; i++)
    ret = ret + "z";
  char buff[512];
  sprintf(buff, "_%.02f_%.02f.nfo", sigma_x, sigma_z);
  ret = ret + buff;
  return ret;
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

vector< vector< double > > allocateMatrix(int rows, int cols)
{
  vector< vector< double > > toRet(rows);
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      toRet[i].push_back(0);
  return toRet;

}

void getMaxInMatrix(vector< vector< double > > & matrix, double& value, int& row, int& col)
{
  value = DBL_MIN;
  for(int i_row = 0; i_row < matrix.size(); i_row++)
    for(int i_col = 0; i_col < matrix[i_row].size(); i_col++)
      if(matrix[i_row][i_col] > value){
        value = matrix[i_row][i_col];
        row = i_row;
        col = i_col;
      }
}

void getMinInMatrix(vector< vector< double > > & matrix, double& value, int& row, int& col)
{
  value = DBL_MAX;
  for(int i_row = 0; i_row < matrix.size(); i_row++)
    for(int i_col = 0; i_col < matrix[i_row].size(); i_col++)
      if(matrix[i_row][i_col] < value){
        value = matrix[i_row][i_col];
        row = i_row;
        col = i_col;
      }
}

void saveMatrix(vector< vector< double > > & matrix, string filename)
{
  std::ofstream out(filename.c_str());
  out << std::setprecision(20) << std::scientific;
  for(int j = 0; j < matrix.size(); j++){
    for(int i = 0; i < matrix[j].size()-1; i++)
      out << matrix[j][i] << " ";
    out << matrix[j][matrix[j].size()-1] << std::endl;
  }
  out.close();
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
  out << std::setprecision(20) << std::scientific;
  for(int i = 0; i < vc.size(); i++)
    out << vc[i] << std::endl;
  out.close();
}

vector< double > readVectorDouble(string filename){
  assert(fileExists(filename));
  std::ifstream in(filename.c_str());
  if(!in.good())
    {
      printf("readVectorDouble::The file %s can not be opened\n",filename.c_str());
      exit(0);
    }
  vector< double > toReturn;
  string s;
  while(getline(in,s))
    {
      stringstream ss(s);
      double d;
      while(!ss.fail()){
        ss >> d;
        if(!ss.fail()){
          toReturn.push_back(d);
        }
      }
    }

  in.close();
  return toReturn;


}


void saveFloatVector( vector< float >& vc, string filename){
  std::ofstream out(filename.c_str());
  for(int i = 0; i < vc.size(); i++)
    out << vc[i] << std::endl;
  out.close();
}

void renderString(const char* format, ...)
{
 va_list args;
 char    buffer[512];
 va_start(args,format);
 vsnprintf(buffer,sizeof(buffer)-1,format,args);
 va_end(args);
 void *font = GLUT_BITMAP_8_BY_13;
 glRasterPos2f(-1,-1);
 for (const char *c=buffer; *c != '\0'; c++) {
   glutBitmapCharacter(font, *c);
 }
}


void secondStatistics(vector< double > data, double* mean, double* variance)
{
  *mean = 0;
  *variance = 0;
  for(int i = 0; i < data.size(); i++)
    *mean = *mean +  data[i];
  *mean = *mean/data.size();

  for(int i = 0; i < data.size(); i++)
    *variance = *variance + (data[i]-*mean)*(data[i]-*mean);
  *variance = sqrt(*variance/data.size());

}


bool directoryExists(string strPath)
{
if ( access( strPath.c_str(), 0 ) == 0 )
    {
        struct stat status;
        stat( strPath.c_str(), &status );

        if ( status.st_mode & S_IFDIR )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
      return false;
    }
}

int makeDirectory(string directory){
  return mkdir(directory.c_str(), 0775);
}


int copyFile(string initialFilePath, string outputFilePath)
{
  ifstream initialFile(initialFilePath.c_str(), ios::in|ios::binary);
  ofstream outputFile (outputFilePath.c_str(),  ios::out|ios::binary);

  initialFile.seekg(0, ios::end);
  outputFile << initialFile.rdbuf();

  /*
  long fileSize = initialFile.tellg();

  if(initialFile.is_open() && outputFile.is_open())
    {
      short * buffer = new short[fileSize/2];
      initialFile.seekg(0, ios::beg);
      initialFile.read((char*)buffer, fileSize);
      outputFile.write((char*)buffer, fileSize);
      delete[] buffer;
    }
  else if(!outputFile.is_open())
    {
      cout<<"I couldn't open "<<outputFilePath<<" for copying!\n";
      return 0;
    }
  else if(!initialFile.is_open())
    {
      cout<<"I couldn't open "<<initialFilePath<<" for copying!\n";
      return 0;
    }
  */
  initialFile.close();
  outputFile.close();
  return 1;
}

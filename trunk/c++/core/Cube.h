/** Cube.cpp
 * defines a cube of voxels. Each voxel would be represented as an array of unsigned bytes. It will be needed to improve it at some point.
The coordinate system would be as in openCV (i.e. top-left). The z direction goes towards the screen. The integral volume that can also
be loaded will share the same coordinate system as the others

Cube/texture coordinates    Cube vertex        World Coordinates
  Z t                     1-------------2      Y
 /                       /|            /|      |  Z
/                       / |           / |      | /
0------ X r            0-------------3  |      |/
|                      |  |          |  |      O-----X
|                      |  5--------- |- 6
|                      | /           | /       Centered in the center of the cube
Y s                    |/            |/
                       4-------------7



*/

#ifndef CUBE_H_
#define CUBE_H_

// #define ulong unsigned long long

#include "neseg.h"
#include "Image.h"
#include "polynomial.h"
#include "utils.h"
#include "VisibleE.h"
#include "Cube_P.h"
#ifdef WITH_OPENMP

#include <omp.h>
#endif

#define CUBE_MAX_X 10000
#define CUBE_MAX_Y 10000
#define CUBE_MAX_Z 1000

using std::string;
using std::vector;
using namespace std;


template <class T, class U>
class Cube : public Cube_P
{

public:

  //Keeps pointers to the ordered (for fast indexing). it goes as voxels[z][y][x]
  T*** voxels;
  U*** voxels_integral;

  //Pointer to the volume data
  T* voxels_origin;
  U* voxels_integral_origin;

  unsigned int wholeTexture;
  unsigned int wholeTextureTrue;
  GLuint wholeTextureDepth;


  int fildes;

  Cube();
  Cube(string filenameParams, bool load_volume_file = true);
  Cube(string filenameParams, string _filenameVoxelData);
  Cube(string filenameParams, string _filenameVoxelData, string filenameIntegralData);


  ~Cube();

  /**********************************************************************
   ** DATA FUNCTIONS. RELATED TO THE CREATION AND HANDLING OF THE DATA **
   **********************************************************************/

  /** Parses the parameter file and loads the cube.*/
  void load_parameters(string filenameParams);

  /** Saves the parameters into the given file.*/
  void save_parameters(string filenameParams);

  /** Initializes the pointers to the volume file*/
  void load_volume_data(string filenameVoxelData);

  /** Initializes the pointers to the integral cube.*/
  void load_integral_volume(string filename);

  void create_volume_file(string filename);

  /** Calculates the integralcube and stores it in filename*/
  void create_integral_cube(string filename);

  /** Calculates the integralcube layer per layer and stores it in filename*/
  void create_integral_cube_by_layers(string filename);

  /** Creates a cube with the trick of the kevin images.*/
  void create_cube_from_kevin_images
  ( string directory, string format, int layer_init, int layer_end,
    float voxelWidth, float voxelHeight, float voxelDepth);

  /** Creates a cube from an image directory.*/
  void create_cube_from_directory
  ( string directory, string format, int layer_init, int layer_end,
    float voxelWidth, float voxelHeight, float voxelDepth, string volume_name = "volume",
    bool invert = true);

  /** Creates a cube from a set of my own "raw" format.*/
  void create_cube_from_raw_files
  ( string directory, string format, int layer_init, int layer_end,
    float voxelWidth, float voxelHeight, float voxelDepth, string volume_name = "volume",
    bool invert = true);
  

  /** Creates a cube from a directory matrix.*/
  void create_cube_from_directory_matrix
  (
   string directory, string format,
   int row_begin, int row_end,
   int col_begin, int col_end,
   int layer_begin, int layer_end,
   float voxelWidth, float voxelHeight, float voxelDepth
   );

  /** Creates a cube from a directory matrix calculating affine transforms between the images of the matrix. The transforms are calculated only in 2D using the MIP's.*/
  void create_cube_from_directory_matrix_with_affine_transformation
  (
   string directory, string format,
   int row_begin, int row_end,
   int col_begin, int col_end,
   int layer_begin, int layer_end,
   float voxelWidth, float voxelHeight, float voxelDepth,
   string name = "volume"
   );

  /** Creates a float cube from a set of images calculated with the "image.h" class.*/
  void create_cube_from_float_images
  (
   string format,
   float idx_begin,
   float idx_end,
   float increment,
   float voxelWidth, float voxelHeight, float voxelDepth,
   string cubeName
   );


  /** Auxiliary enum used by create_cube_from_directory_matrix_with_affine_transformation.*/
  typedef enum {
    STITCH_LEFT, STITCH_UP, STITCH_LEFTUP
  } WhereToCorrelate;


  /** Auxiliary method used by the create_cube_from_directory_matrix_with_affine_transformation. Calculates an affine transformation of orig, stores the result in dest and puts to 1 the values of mask.*/
  void apply_affine_transform
  (gsl_matrix* transform,
   IplImage*   orig,
   IplImage*   dest,
   vector< vector< int > >* mask
   );

  /** Auxiliary method used by the create_cube_from_directory_matrix_with_affine_transformation. Calculates the cross correlation of the stitched image and the img, using mask to know where are valid pixels. It applies the ey and ex offsets.*/
  double cross_correlate
  ( gsl_matrix* trns_init,
    IplImage* stitched,
    vector<vector< int > >& mask,
    IplImage* img,
    int ey,
    int ex,
    WhereToCorrelate where
    );

  /** Auxiliary method used by the create_cube_from_directory_matrix_with_affine_transformation. Calculates the registration of one image with the stitched one using cross correlation over the left, up or both margins..*/
  gsl_matrix* find_affine_transform_from_correlation
  ( gsl_matrix* trns_init,
    IplImage* stitched,
    vector<vector< int > >& mask,
    IplImage* img,
    WhereToCorrelate where
    );


  /** Creates the MIP projections of the images on the matrix directory.*/
  void create_directory_matrix_MIP
  (
   string directory, string format, 
   int row_begin, int row_end,
   int col_begin, int col_end,
   int layer_begin, int layer_end
   );

  /** Saves the cube as a stack of images.*/
  void save_as_image_stack(string filename = "");

  /** Creates the MIP image of the cube.*/
  void createMIPImage(string filename = "", bool minMax = 0); //0 stands for minimum intensity projection
  
  /** Converts from coordinates in micrometers to a position in indexes.*/
  void micrometersToIndexes(vector<float>& micrometers, vector< int >& indexes);

  /** Converts from coordinates in cube indexes to micrometers.*/
  void indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers);

  /** Print some cube statistics.*/
  void print_statistics(string filename = "");

  /** Outputs the histogram of gray values into a file.*/
  void histogram(string filename = "");

  /** Ouputs the histogram of gray values into a file, but ignores voxels with value 0.*/
  void histogram_ignoring_zeros(string filename = "");

  /** Outputs a volume with valoues only in places where the mask is not 0.*/
  void apply_mask(string mask_nfo, string mask_vl, string output_nfo, string output_vl);

  /** Calculates the integral allong the line betwewn  the indexes.*/
  double integral_between(int x0, int y0, int z0, int x1, int y1, int z1);

  /** Prints the parameters*/
  void print();

  /** Returns the value of the voxel at indexes x,y,z.*/
  T at(int x, int y, int z);

  /** Changes the value of the voxel at indexes x,y,z.*/
  void put(int x, int y, int z, T value);

  /** Puts all the voxels to a given value.*/
  void put_all(T value);

  /** Returns the value of the integral volume at indexes x,y,z.*/
  U integral_volume_at(int x, int y, int z);

  /** Cuts the cube from the first point to the second point.*/
  void cut_cube(int x0, int y0, int z0, int x1, int y1, int z1, string name);

  void print_size();

  /** Duplicates the cube and puts it all in 0.*/
  Cube<T,U>* duplicate_clean(string filename);

  /** Creates a blank cube with the same dimensions and float type.*/
  Cube<float,double>* create_blank_cube(string filename);

  /** Creates a blanck cube with the same dimensions and uchar type*/
  Cube<uchar,ulong>*  create_blank_cube_uchar(string filename);

  /** Creates a gaussian pyramid of the cube.*/
  void create_gaussian_pyramid();

  /** Creates a gaussian pyramid of the cube. Do not subsample in the Z direction.*/
  void create_gaussian_pyramid_2D();


  /**********************************************************************
   **           DISPLAY FUNCTIONS                                      **
   **********************************************************************/

  /** Subsamples the volume and loads it as a texture.*/
  void load_whole_texture();

  /** Loads the texture of the brick represented by the row, col. Both start at 0.*/
  void load_texture_brick(int row, int col);

  /** Loads a thresholded brick.*/
  void load_thresholded_texture_brick(int row, int col, float threshold);

  /** Loads a doubly thresholded brick using floats.*/ 
  void load_thresholded_maxmin_texture_brick_float
    (int row, int col, float threshold_low = -1e6, float threshold_high = 1e6);

  /** Loads a thresholded brick using floats.*/
  void load_thresholded_texture_brick_float(int row, int col, float threshold);

  /** Draws a set of planes with the texture among them.*/
  void draw_layers_parallel();

  /** Draws a MIP projection of the cube.*/
  void draw(float rotx, float roty, float nPlanes = 200.0, int min_max = 0, int microm_voxels = 1);

  /** Draws the whole cube.*/
  void draw_whole(float rotx, float roty, float nPlanes, int min_max);

  /** Draws the cube between the points given.*/
  void draw
    ( int x0, int y0, int z0, int x1, int y1, int z1, float rotx, float roty, 
      float nPlanes, int min_max, float threshold = -1e6);

  /** Draws the XY layer number depth of the brick.*/
  void draw_layer_tile_XY(float depth, int color = 0);

  /** Draws the XZ layer number depth of the brick.*/
  void draw_layer_tile_XZ(float depth, int color = 0);

  /** Draws the YZ layer number depth of the brick.*/
  void draw_layer_tile_YZ(float depth, int color = 0);

  /** Draws an orientation grid for the cube.*/
  void draw_orientation_grid(bool include_split = true);

  /** Renders a string.*/
  void render_string(const char* format, ...);

  /** Draws the cube in micrometers.*/
  void draw();



  /**********************************************************************
   **           AUXILIARY MATHEMATICAL FUNCTIONS                       **
   **********************************************************************/


  /** Inverts a matrix in the opengl format. Returns also opengl format.*/
  GLfloat* invert_matrix(GLfloat* a);

  /** Product between a matrix and a vector.*/
  GLfloat* matrix_vector_product(GLfloat* matrix, GLfloat* vector);

  /** Infers the rotation angles from a Matrix. Not working.*/
  GLfloat* get_matrix_angles(GLfloat* m);

  /** Creates a vector with values x, y, z.*/
  GLfloat* create_vector(float x, float y, float z, float w);

  vector< T > sort_values();

  int find_value_in_ordered_vector(vector< T >& vector, T value);




  /**********************************************************************
   **           IMAGE OPERATIONS DONE IN 3D IN THE CUBE                **
   **********************************************************************/

  /** Downsamples the cube in a naive manner.*/
  void subsampleMean(string dir_name);

  /** Downsamples the cube. Does not work, do not even try it.*/
  void subsampleMinimum();

  /** Substract the mean value to the cube.*/
  void substract_mean(string name);

  /** Creates two gaussian masks. The code is taken from Libvision.*/
  int gaussian_mask(float sigma, vector< float >& Mask0, vector< float >& Mask1);

  /** Creates two masks, one with a regular gaussian and the other with the second derivate of the gaussian.*/
  int gaussian_mask_second(float sigma, vector< float >& Mask0, vector< float >& Mask1);

  /** Convolves the cube with a mask horizontally. Saves the result in output.*/
  void convolve_horizontally(vector< float >& Mask, Cube<float,double>* output, bool use_borders = true);

  /** Convolves the cube with a mask vertically. Saves the result in output.*/
  void convolve_vertically(vector< float >& Mask, Cube<float,double>* output, bool use_borders = true);

  /** Convolves the cube with a mask in depth. Saves the result in output.*/
  void convolve_depth(vector< float >& Mask, Cube<float,double>* output, bool use_borders = true);

  /**  Finds the gradient in the x direction.*/
  void gradient_x
  (float sigma_xy, float sigma_z,
   Cube< float,double >* output, Cube<float,double>* tmp);

  /**  Finds the gradient in the y direction.*/
  void gradient_y
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /**  Finds the gradient in the z direction.*/
  void gradient_z
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Finds the second derivate in the x direction.*/
  void second_derivate_xx
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Finds the second derivate in the y direction.*/
  void second_derivate_yy
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Finds the second derivate in the z direction.*/
  void second_derivate_zz
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Finds the second derivate in the xy direction.*/
  void second_derivate_xy
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Finds the second derivate in the xz direction.*/
  void second_derivate_xz
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Finds the second derivate in the yz direction.*/
  void second_derivate_yz
  (float sigma_xy, float sigma_z,
   Cube<float,double>* output, Cube<float,double>* tmp);

  /** Blurs the cube with the variance sigma.*/
  void blur(float sigma,
            Cube<float, double>* output,
            Cube<float,double>* tmp);

  /** Blurs the cube in 2D with the variance sigma.*/
  void blur_2D(float sigma,
            Cube<float, double>* output,
            Cube<float,double>* tmp);

  /** Calculates the derivative in the orders defined.*/
  void calculate_derivative
  (int nx, int ny, int nz,
   float sigma_x, float sigma_y, float sigma_z,
   Cube<float, double>* output, Cube<float, double>* tmp);

  /** Calculates all the second derivates of the cube.*/
  void calculate_second_derivates(float sigma_xy, float sigma_z);

  /** Calculates the eigenvalues of the cube using the information in the directory name.*/
  void calculate_eigen_values(string directory_name);

  /** Calculates the eigenvalues of the cube using the information in the directory name.*/
  void calculate_eigen_values(float sigma_xy, float sigma_z, bool calculate_eigen_vectors);

  /** Calculates the eigenvectors of the cube using the information in the directory name.*/
  void calculate_eigen_vector_lower_eigenvalue(string directory_name);

  /** Calculates the f-measure of the cube using the information in the directory name.*/
  void calculate_f_measure(float sigma_xy, float sigma_z);

  /** Calculates the measure of Aguet05.*/
  void calculate_aguet(float sigma_xy, float sigma_z = 0);
  void calculate_aguet_flat(float sigma_xy, float sigma_z = 0);

  /** Creates three new volumes with the eigenvalues ordered.*/
  void order_eigen_values(float sigma_xy, float sigma_z);

  /** Produces a vector of indexes with local maxima of the cube.*/
  vector< vector< int > > decimate
   (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
    bool save_boosting_response = false);

  /** Produces a vector of indexes with local maxima of the cube.*/
  vector< vector< int > > decimate_log
  (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
     bool save_boosting_response = false);

  /** Produces a vector of the NMS in the layer indicated.*/
  vector< vector< int > > decimate_layer
  (int nLayer, float threshold, int window_xy, string filename);

  /** Renders a cylinder between the points in v1 and v2. Its values will be put to 1.*/
  void render_cylinder(vector<int> idx1, vector<int> idx2, float radius_micrometers);

  void norm_cube(Cube<float,double>* c1, Cube<float,double>* c2, 
                 Cube<float,double>* output);

  /** Finds the maximum and the minimum of a cube.*/
  void min_max(float* min, float* max);

  void norm_cube
  (string volume_nfo, string volume_1, string volume_2,
   string volume_3, string volume_output);

  void get_ROC_curve
  (string volume_nfo, string volume_positive, string volume_negative,
   string output_file = "ROC.txt", int nPoints = 100);

  Cube_P* threshold(float thres, string outputName="output");

};


//######### CUBE MAIN #####################


template <class T, class U>
Cube<T,U>::Cube()
{
  fildes = -1;
  filenameVoxelData = "";
}
template <class T, class U>
Cube<T,U>::~Cube()
{
  if(fildes != -1){
    munmap(voxels_origin, cubeWidth*cubeHeight*cubeDepth*sizeof(T));
    close(fildes);
  }
}

template <class T, class U>
Cube<T,U>::Cube(string filenameParams, string _filenameVoxelData)
{
  fildes = -1;
  filenameVoxelData = "";

  if (filenameParams!= "")
    load_parameters(filenameParams);

  if (_filenameVoxelData!= "")
    load_volume_data(_filenameVoxelData);

  nColToDraw = -1;
  nRowToDraw = -1;
  glGenTextures(1, &wholeTexture);
  glGenTextures(1, &wholeTextureTrue);


}

template <class T, class U>
Cube<T,U>::Cube(string filenameParams, string filenameVoxelData, string filenameIntegralData)
{
  fildes = -1;
  if (filenameParams!= "")
    load_parameters(filenameParams);

  if (filenameVoxelData!= "")
    load_volume_data(filenameVoxelData);

  if(filenameIntegralData != "")
    load_integral_volume(filenameIntegralData);

  nColToDraw = -1;
  nRowToDraw = -1;
  glGenTextures(1, &wholeTexture);
  glGenTextures(1, &wholeTextureTrue);
}

template <class T, class U>
Cube<T,U>::Cube(string filenameParams, bool load_volume_file)
{
//   string filenameParams = dirName + "/volume.nfo";
//   string filenameVoxelData = dirName + "/volume.vl";
//   string filenameIntegralData = dirName + "/volume.iv";

//   if (filenameParams!= "")
//     load_parameters(filenameParams);

//   if (filenameVoxelData!= "")
//     load_volume_data(filenameVoxelData);

//   if(filenameIntegralData != "")
//     load_integral_volume(filenameIntegralData);

//   nColToDraw = -1;
//   nRowToDraw = -1;
//   glGenTextures(1, &wholeTexture);
//   glGenTextures(1, &wholeTextureTrue);

  fildes = -1;
  load_parameters(filenameParams);

  if(load_volume_file){
    string directory = filenameParams.substr(0,filenameParams.find_last_of("/\\")+1); 
    if((filenameVoxelData != "")){
      load_volume_data(directory + filenameVoxelData);
    }
  }

  nColToDraw = -1;
  nRowToDraw = -1;
  glGenTextures(1, &wholeTexture);
  glGenTextures(1, &wholeTextureTrue);

}

template <class T, class U>
void Cube<T,U>::print_size()
{
  printf("The size of the template is %i and the type %s\n", sizeof(T), type.c_str());
}

/** Functions related to handling the data in the cube.*/
#include "Cube_data.h"

/** Cube display functions. */
#include "Cube_display.h"

/** Auxiliary mathematical functions. */
#include "Cube_aux.h"

/** Image operations in the cube.*/
#include "Cube_image.h"






#endif

template <class T, class U>
void Cube<T,U>::load_parameters(string filenameParams)
 {

   fileParams = filenameParams;

   filenameVoxelData = "";

   directory = getDirectoryFromPath(filenameParams);

   filenameParameters = getNameFromPath(filenameParams);


   std::ifstream file(filenameParams.c_str());
   if(!file.good())
     printf("Cube<T,U>::load_parameters: error loading the file %s\n", filenameParams.c_str());

   string name;
   string attribute;
   while(file.good())
     {
       file >> name;
       file >> attribute;
       if(!strcmp(name.c_str(), "voxelDepth"))
         voxelDepth = atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "voxelHeight"))
         voxelHeight = atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "voxelWidth"))
         voxelWidth =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeDepth"))
         cubeDepth = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeHeight"))
         cubeHeight = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeWidth"))
         cubeWidth = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "x_offset"))
         x_offset =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "y_offset"))
         y_offset =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "z_offset"))
         z_offset =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeFile"))
         filenameVoxelData = attribute;
       else if(!strcmp(name.c_str(), "type"))
         type = attribute;
       else
         printf("Cube<T,U>::load_parameters: Attribute %s and value %s not known\n", name.c_str(), attribute.c_str());
     }

   //This is usually because it is an old cube file (has no type and
   //no indication of where is the cubeFile)
   if(filenameVoxelData == ""){
     filenameVoxelData = getNameFromPathWithoutExtension(filenameParams) + ".vl";
     int size = getFileSize(directory + filenameVoxelData);
     if(  size == cubeDepth*cubeHeight*cubeWidth)
       type = "uchar";
     else if ( size == 4*cubeDepth*cubeHeight*cubeWidth)
       type = "float";
     else{
       printf("Unable to guess the type inside the file. Exiting\n");
       exit(0);
     }
   }


   //  #if debug
   printf("Cube parameters:\n");
   printf("  directory %s, parameters %s\n", directory.c_str(), filenameParameters.c_str());
   printf("  cubeWidth %i cubeHeight %i cubeDepth %i\n", cubeWidth, cubeHeight, cubeDepth);
   printf("  voxelWidth %f voxelHeight %f voxelDepth %f\n", voxelWidth, voxelHeight, voxelDepth);
   printf("  x_offset %f y_offset %f z_offset %f\n", x_offset, y_offset, z_offset);
   //  #endif
 }


template <class T, class U>
void Cube<T,U>::save_parameters(string filename)
{
  std::ofstream out(filename.c_str());
  out << "voxelDepth "  << voxelDepth  << std::endl;
  out << "voxelHeight " << voxelHeight << std::endl;
  out << "voxelWidth "  << voxelWidth  << std::endl;
  out << "cubeDepth "   << cubeDepth  << std::endl;
  out << "cubeHeight "  << cubeHeight << std::endl;
  out << "cubeWidth "   << cubeWidth  << std::endl;
  out << "x_offset "    << x_offset   << std::endl;
  out << "y_offset "    << y_offset   << std::endl;
  out << "z_offset "    << z_offset   << std::endl;
  out << "cubeFile "    << filenameVoxelData << std::endl;
  out << "type "        << type << std::endl;


  out.close();
}


template <class T, class U>
void Cube<T,U>::load_volume_data(string filenameVoxelData)
{
 fildes = open64(filenameVoxelData.c_str(), O_RDWR);

 if(fildes == -1) //The file does not exist
   {
     printf("The file %s does not exist.Aborting.\n", filenameVoxelData.c_str());
     exit(0);
   }

 void* mapped_file = mmap64(0,
                            cubeWidth*cubeHeight*cubeDepth*sizeof(T),
                            PROT_READ|PROT_WRITE, MAP_SHARED, fildes, 0);

 if(mapped_file == MAP_FAILED)
    {
      printf("Cube<T,U>::load_volume_data: There is a bug here, volume not loaded. %s\n", 
              filenameVoxelData.c_str());
      exit(0);
    }

  voxels_origin = (T*) mapped_file;
  voxels = (T***)malloc(cubeDepth*sizeof(T**));

  //Initializes the pointer structure to acces quickly to the voxels
  for(int z = 0; z < cubeDepth; z++){
      voxels[z] = (T**)malloc(cubeHeight*sizeof(T*));
      for(int j = 0; j < cubeHeight; j++){
          voxels[z][j]=(T*)&voxels_origin[z*cubeWidth*cubeHeight + j*cubeWidth];
        }
    }
}

template <class T, class U>
void Cube<T,U>::load_integral_volume(string filename)
{
 int fildes = open64(filename.c_str(), O_RDWR);
 void* mapped_file = mmap64(0,cubeWidth*cubeHeight*cubeDepth*sizeof(U), PROT_READ, MAP_PRIVATE, fildes, 0);
 if(mapped_file == MAP_FAILED)
    {
      printf("Cube<T,U>::load_integral_volume(%s): Volume not loaded\n", filename.c_str());
      exit(0);
    }
  voxels_integral_origin = (U*) mapped_file;
  voxels_integral = (U***)malloc(cubeDepth*sizeof(U**));

  //Initializes the pointer structure to acces quickly to the voxels
  for(int z = 0; z < cubeDepth; z++)
    {
      voxels_integral[z] = (U**)malloc(cubeHeight*sizeof(U*));
      for(int j = 0; j < cubeHeight; j++)
        {
          voxels_integral[z][j]=(U*)&voxels_integral_origin[z*cubeWidth*cubeHeight + j*cubeWidth];
        }
    }
}


//FIXME
template <class T, class U>
void Cube<T,U>::create_volume_file(string filename)
{

  printf("Creating volume file in %s\n", filename.c_str());
  FILE* fp = fopen(filename.c_str(), "w");
  int line_length = 0;
//   if ((colOffset == 0) && (rowOffset == 0))
//     line_length = cubeWidth;
//   else
  line_length = cubeWidth;
  //FIXME
  T buff[line_length];
  for(int i = 0; i < line_length; i++)
    buff[i] = 0;

  for(int i = 0; i < cubeHeight*cubeDepth; i++)
    {
      int err = fwrite(buff, sizeof(T), line_length, fp);
      if(err == 0)
        printf("Cube::create_volume_file(%s): error writing the layer %i\n", filename.c_str(), i);
    }
  fclose(fp);
}


template <class T, class U>
void Cube<T,U>::create_integral_cube(string filename)
{
  char buff[1024];
  sprintf(buff, "touch %s", filename.c_str());
  system(buff);
  printf("Cube::create_integral_cube(): creating the file %s\n", filename.c_str());
  int fp = open64(filename.c_str(), O_WRONLY || O_SYNC || O_LARGEFILE );
  if(fp == -1)
    {
      printf("Cube::create_integral_cube(): error openning the file %s\n", filename.c_str());
      exit(0);
    }

  ulong* temporal_layer_integral = (ulong*)malloc(cubeWidth*cubeHeight*sizeof(ulong));
  for(int i = 0; i < cubeHeight; i++)
    for(int j = 0; j < cubeWidth; j++)
        temporal_layer_integral[i*cubeWidth + j] = 0;

  int* temporal_layer_depth = (int*)malloc(cubeWidth*cubeHeight*sizeof(int));
  for(int i = 0; i < cubeHeight; i++)
    for(int j = 0; j < cubeWidth; j++)
        temporal_layer_depth[i*cubeWidth + j] = 0;

  printf("Calculating the integral cube: [");
  ulong accumulator = 0;

  for(int depth = 0; depth < cubeDepth; depth++)
    {
      //Calculates the first row
      accumulator = 0;
      for(int col = 0; col < cubeWidth; col++){
        accumulator += voxels[depth][0][col];
        temporal_layer_integral[col] += accumulator;
        temporal_layer_depth[col] += voxels[depth][0][col];
      }

      accumulator = voxels[depth][0][0];
      for(int row = 1; row < cubeHeight; row++){
        accumulator += voxels[depth][row][0];
        temporal_layer_integral[row*cubeWidth] += accumulator;
        temporal_layer_depth[row*cubeWidth] += voxels[depth][row][0];
      }

      //Calculates the rest of the rows
      for(int row = 1; row < cubeHeight; row++){
        for(int col = 1; col < cubeWidth; col++){
          temporal_layer_depth[col + row*cubeWidth] += voxels[depth][row][col];
          temporal_layer_integral[col + row*cubeWidth] =
            temporal_layer_depth[col + row*cubeWidth] +
            temporal_layer_integral[col + (row-1)*cubeWidth] -
            temporal_layer_integral[col-1 + (row-1)*cubeWidth] +
            temporal_layer_integral[col-1 + row*cubeWidth];
        }
      }
      printf("#");
      fflush(stdout);
//       printf("%lu\n", temporal_layer_integral[cubeWidth*cubeHeight-1]);

      //Saves the first image in the volum2
      int err = write(fp, temporal_layer_integral, cubeHeight*cubeWidth*sizeof(ulong));
      if(err == -1)
        printf("Cube::create_integral_cube(%s): error writing the layer %i\n", filename.c_str(), depth);
    }
  printf("]\n");
  close(fp);
}

template <class T, class U>
void Cube<T,U>::create_integral_cube_by_layers(string filename)
{
  char buff[1024];
  sprintf(buff, "touch %s", filename.c_str());
  system(buff);
  printf("Cube::create_integral_cube(): creating the file %s\n", filename.c_str());
  int fp = open64(filename.c_str(), O_WRONLY || O_SYNC || O_LARGEFILE );
  if(fp == -1)
    {
      printf("Cube::create_integral_cube(): error openning the file %s\n", filename.c_str());
      exit(0);
    }

  ulong* temporal_layer = (ulong*)malloc(cubeWidth*cubeHeight*sizeof(ulong));
  for(int i = 0; i < cubeHeight; i++)
    for(int j = 0; j < cubeWidth; j++)
        temporal_layer[i*cubeWidth + j] = 0;

  printf("Calculating the integral cube: [");
  ulong accumulator = 0;

  for(int depth = 0; depth < cubeDepth; depth++)
    {
      //Calculates the first row
      accumulator = 0;
      for(int col = 0; col < cubeWidth; col++){
        accumulator += voxels[depth][0][col];
        temporal_layer[col] = accumulator;
      }

      //Calculates the first column
      accumulator = voxels[depth][0][0];
      for(int row = 1; row < cubeHeight; row++){
        accumulator += voxels[depth][row][0];
        temporal_layer[row*cubeWidth] = accumulator;
      }

      //Calculates the rest of the rows
      for(int row = 1; row < cubeHeight; row++){
        for(int col = 1; col < cubeWidth; col++){
          temporal_layer[col + row*cubeWidth] =
            voxels[depth][row][col] +
            temporal_layer[col + (row-1)*cubeWidth] -
            temporal_layer[col-1 + (row-1)*cubeWidth] +
            temporal_layer[col-1 + row*cubeWidth];
        }
      }
      printf("#");
      fflush(stdout);

      //Saves the first image in the volum2
      int err = write(fp, temporal_layer, cubeHeight*cubeWidth*sizeof(ulong));
      if(err == -1)
        printf("Cube::create_integral_cube(%s): error writing the layer %i\n",
               filename.c_str(), depth);
    }
  printf("]\n");
  close(fp);
}

// template <class T, class U>
// void Cube<T,U>::create_integral_cube_float(string filename)
// {
//   printf("Cube::create_integral_cube(): creating the file %s\n", filename.c_str());
//   int fp = open64(filename.c_str(), O_WRONLY || O_SYNC || O_LARGEFILE );
//   if(fp == -1)
//     {
//       printf("Cube::create_integral_cube(): error openning the file %s\n", filename.c_str());
//       exit(0);
//     }

//   double* temporal_layer_integral = (double*)malloc(cubeWidth*cubeHeight*sizeof(double));
//   for(int i = 0; i < cubeHeight; i++)
//     for(int j = 0; j < cubeWidth; j++)
//         temporal_layer_integral[i*cubeWidth + j] = 0;

//   double* temporal_layer_depth = (double*)malloc(cubeWidth*cubeHeight*sizeof(double));
//   for(int i = 0; i < cubeHeight; i++)
//     for(int j = 0; j < cubeWidth; j++)
//         temporal_layer_depth[i*cubeWidth + j] = 0;

//   printf("Calculating the integral cube: [");
//   double accumulator = 0;

//   for(int depth = 0; depth < cubeDepth; depth++)
//     {
//       //Calculates the first row
//       accumulator = 0;
//       for(int col = 0; col < cubeWidth; col++){
//         accumulator += voxels[depth][0][col];
//         temporal_layer_integral[col] += accumulator;
//         temporal_layer_depth[col] += voxels[depth][0][col];
//       }

//       accumulator = voxels[depth][0][0];
//       for(int row = 1; row < cubeHeight; row++){
//         accumulator += voxels[depth][row][0];
//         temporal_layer_integral[row*cubeWidth] += accumulator;
//         temporal_layer_depth[row*cubeWidth] += voxels[depth][row][0];
//       }

//       //Calculates the rest of the rows
//       for(int row = 1; row < cubeHeight; row++){
//         for(int col = 1; col < cubeWidth; col++){
//           temporal_layer_depth[col + row*cubeWidth] += voxels[depth][row][col];
//           temporal_layer_integral[col + row*cubeWidth] =
//             temporal_layer_depth[col + row*cubeWidth] +
//             temporal_layer_integral[col + (row-1)*cubeWidth] -
//             temporal_layer_integral[col-1 + (row-1)*cubeWidth] +
//             temporal_layer_integral[col-1 + row*cubeWidth];
//         }
//       }
//       printf("#");
//       fflush(stdout);
// //       printf("%lu\n", temporal_layer_integral[cubeWidth*cubeHeight-1]);

//       //Saves the first image in the volum2
//       int err = write(fp, temporal_layer_integral, cubeHeight*cubeWidth*sizeof(double));
//       if(err == -1)
//         printf("Cube::create_integral_cube(%s): error writing the layer %i\n", filename.c_str(), depth);
//     }
//   printf("]\n");
//   close(fp);
// }



template <class T, class U>
void Cube<T,U>::create_cube_from_kevin_images(
        string directory, string format, int begin, int end,
        float voxelWidth_p, float voxelHeight_p, float voxelDepth_p)
{
  string name = directory + "/volume.vl";
  FILE *fp = fopen(name.c_str(), "w+");
  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];
  printf("Generating the cube: [");
//   cvNamedWindow("pepe",1);
  int cubeWidth_p = 0;
  int cubeHeight_p = 0;
  for(int z = begin; z <= end; z++)
    {
      printf("#");
      fflush(stdout);
      sprintf(buff, image_format, z);
      IplImage* pepe = cvLoadImage(buff,0);
      cubeWidth_p = pepe->width;
      cubeHeight_p = pepe->height;
      IplImage* pepe_low = cvCreateImage(cvSize(pepe->width, pepe->height), IPL_DEPTH_8U, 1);
      for(int y = 0; y < pepe_low->height; y++)
        for(int x = 0; x < pepe_low->width; x++)
          pepe_low->imageData[x + y*pepe_low->widthStep] = (T)pepe->imageData[x + y*pepe->widthStep]*32;
//       cvShowImage("pepe", pepe_low);
//       cvWaitKey(1000);
//       if(z == 20)
//         cvSaveImage("kevin.jpg", pepe_low);
      for(int y = 0; y < pepe_low->height; y++)
        fwrite( ((T*)(pepe_low->imageData + pepe_low->widthStep*y )), sizeof(T), pepe_low->width, fp);
      cvReleaseImage(&pepe);
      cvReleaseImage(&pepe_low);
    }

  string parameters_file = directory + "/volume.nfo";
  std::ofstream out_w(parameters_file.c_str());
  out_w << "parentCubeWidth " << cubeWidth_p << std::endl;
  out_w << "parentCubeHeight " << cubeHeight_p << std::endl;
  out_w << "parentCubeDepth " << end - begin + 1 << std::endl;
  out_w << "cubeWidth " << cubeWidth_p << std::endl;
  out_w << "cubeHeight " << cubeHeight_p << std::endl;
  out_w << "cubeDepth " << end - begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth_p << std::endl;
  out_w << "voxelHeight " << voxelHeight_p << std::endl;
  out_w << "voxelDepth " << voxelDepth_p << std::endl;
  out_w << "rowOffset  0" << std::endl;
  out_w << "colOffset  0" << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w.close();

  this->cubeWidth   = cubeWidth_p     ;
  this->cubeHeight  = cubeHeight_p    ;
  this->cubeDepth   = end - begin + 1 ;
  this->voxelWidth  = voxelWidth_p    ;
  this->voxelHeight = voxelHeight_p   ;
  this->voxelDepth  = voxelDepth_p    ;

  printf("]\n");
  fclose(fp);
  load_volume_data(name);
}


template <class T, class U>
void Cube<T,U>::create_cube_from_directory
(string directory, string format,
 int layer_begin, int layer_end,
 float voxelWidth, float voxelHeight, float voxelDepth,
 string volume_name, bool invert)
{

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];

  sprintf(buff, image_format, layer_begin);
  printf("Loading the image %s to get the info of the cube\n", buff);
  IplImage* pepe = cvLoadImage(buff,0);

  string parameters_file = directory + volume_name + ".nfo";
  printf("Saving the parameters file in %s\n", parameters_file.c_str());
  std::ofstream out_w(parameters_file.c_str());
  out_w << "cubeWidth " << pepe->width << std::endl;
  out_w << "cubeHeight " << pepe->height << std::endl;
  out_w << "cubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth << std::endl;
  out_w << "voxelHeight " << voxelHeight << std::endl;
  out_w << "voxelDepth " << voxelDepth << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w << "cubeFile " << "./" + volume_name + ".vl" << std::endl;
  out_w << "type uchar" << std::endl;
  out_w.close();

  this->cubeWidth   = pepe->width     ;
  this->cubeHeight  = pepe->height    ;
  this->cubeDepth   = layer_end - layer_begin + 1 ;
  this->voxelWidth  = voxelWidth    ;
  this->voxelHeight = voxelHeight   ;
  this->voxelDepth  = voxelDepth    ;

  string name = directory + "/" + volume_name + ".vl";
  printf("Saving the raw data in %s\n", name.c_str());
  FILE *fp = fopen(name.c_str(), "w+");
  printf("Generating the cube: [");
//   cvNamedWindow("pepe",1);
  for(int z = layer_begin; z <= layer_end; z++)
    {
      // printf("#");
      fflush(stdout);
      sprintf(buff, image_format, z);
      printf("Adding image: %s\n", buff);
      IplImage* pepe = cvLoadImage(buff,0);
      IplImage* pepe_low = cvCreateImage(cvSize(pepe->width, pepe->height), IPL_DEPTH_8U, 1);
      for(int y = 0; y < pepe_low->height; y++)
        for(int x = 0; x < pepe_low->width; x++)
          if(invert){
            pepe_low->imageData[x + y*pepe_low->widthStep] = 
              255-(T)pepe->imageData[x + y*pepe->widthStep];}
          else {
            pepe_low->imageData[x + y*pepe_low->widthStep] = 
              (T)pepe->imageData[x + y*pepe->widthStep];}
//       cvShowImage("pepe", pepe_low);
//       cvWaitKey(1000);
//       if(z == 20)
//         cvSaveImage("kevin.jpg", pepe_low);
      for(int y = 0; y < pepe_low->height; y++)
        fwrite( ((T*)(pepe_low->imageData + pepe_low->widthStep*y )), sizeof(T), pepe_low->width, fp);
      cvReleaseImage(&pepe);
      cvReleaseImage(&pepe_low);
    }
  // printf("]\n");
  fclose(fp);
  load_volume_data(name);
}


template <class T, class U>
void Cube<T,U>::create_cube_from_raw_files
(string directory, string format,
 int layer_begin, int layer_end,
 float voxelWidth, float voxelHeight, float voxelDepth,
 string volume_name, bool invert)
{

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];

  sprintf(buff, image_format, layer_begin);
  printf("Loading the image %s to get the info of the cube\n", buff);
  Image< float>* pepe = new Image<float>(buff);

  string parameters_file = directory + volume_name + ".nfo";
  printf("Saving the parameters file in %s\n", parameters_file.c_str());
  std::ofstream out_w(parameters_file.c_str());
  out_w << "cubeWidth " << pepe->width << std::endl;
  out_w << "cubeHeight " << pepe->height << std::endl;
  out_w << "cubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth << std::endl;
  out_w << "voxelHeight " << voxelHeight << std::endl;
  out_w << "voxelDepth " << voxelDepth << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w << "cubeFile " << "./" + volume_name + ".vl" << std::endl;
  out_w << "type float" << std::endl;
  out_w.close();

  this->cubeWidth   = pepe->width     ;
  this->cubeHeight  = pepe->height    ;
  this->cubeDepth   = layer_end - layer_begin + 1 ;
  this->voxelWidth  = voxelWidth    ;
  this->voxelHeight = voxelHeight   ;
  this->voxelDepth  = voxelDepth    ;

  string name = directory + "/" + volume_name + ".vl";
  this->create_volume_file(name);
  printf("Saving the raw data in %s\n", name.c_str());
  this->load_volume_data(name);

  for(int z = layer_begin; z <= layer_end; z++)
    {
      fflush(stdout);
      sprintf(buff, image_format, z);
      printf(" -> adding image: %s multiplied by %f\n", buff,
             (float)z*z*z*z);
      Image< float>* pepe = new Image<float>(buff);
      for(int x = 0; x < pepe->width; x++)
        for(int y = 0; y < pepe->height; y++)
          this->put(x,y,z-layer_begin,
                    pepe->at(x,y)*z*z*z*z);
    }
}



template <class T, class U>
void Cube<T,U>::create_cube_from_directory_matrix
(
 string directory, string format,
 int row_begin, int row_end,
 int col_begin, int col_end,
 int layer_begin, int layer_end,
 float voxelWidth, float voxelHeight,
 float voxelDepth
)
{

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];
  sprintf(buff, image_format, layer_begin);

  sprintf(buff, image_format, row_begin, col_begin, layer_begin);
  IplImage* pepe = cvLoadImage(buff,0);

  string parameters_file = directory + "/volume.nfo";
  std::ofstream out_w(parameters_file.c_str());
  out_w << "cubeWidth " << pepe->width*(col_end - col_begin +1) << std::endl;
  out_w << "cubeHeight " << pepe->height*(row_end - row_begin +1) << std::endl;
  out_w << "cubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth << std::endl;
  out_w << "voxelHeight " << voxelHeight << std::endl;
  out_w << "voxelDepth " << voxelDepth << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w << "type  uchar" << std::endl;
  out_w << "cubeFile volume.vl" << std::endl;
  out_w.close();

  this->cubeWidth   = pepe->width*(col_end - col_begin +1)     ;
  this->cubeHeight  = pepe->height*(row_end - row_begin +1)    ;
  this->cubeDepth   = layer_end - layer_begin + 1 ;
  this->voxelWidth  = voxelWidth    ;
  this->voxelHeight = voxelHeight   ;
  this->voxelDepth  = voxelDepth    ;

  // Cube<float, double>* toPut = create_blank_cube();


  string name = directory + "/volume.vl";
  FILE *fp = fopen(name.c_str(), "w+");
  printf("Generating the cube: [");
//   cvNamedWindow("pepe",1);
  for(int z = layer_begin; z <= layer_end; z++)
    {
      for(int y = row_begin; y <= row_end; y++)
        {
          IplImage* row_imgs[col_end - col_begin + 1];
          for(int x = 0; x < col_end - col_begin + 1; x++)
            {
              sprintf(buff, image_format, y,x+col_begin,z);
              row_imgs[x] = cvLoadImage(buff,0);
            }
          for(int row = 0; row < row_imgs[0]->height; row++)
            for(int col = 0; col < col_end - col_begin + 1; col ++)
              fwrite( ((T*)(row_imgs[col]->imageData + row_imgs[col]->widthStep*row )), sizeof(T), row_imgs[col]->width, fp);

          for(int x = 0; x < col_end - col_begin + 1; x++)
            cvReleaseImage(&row_imgs[x]);
        }
      printf("#");
      fflush(stdout);
    }
  printf("]\n");
  fclose(fp);
  load_volume_data(name);
}


template <class T, class U>
void Cube<T,U>::create_directory_matrix_MIP
(
 string directory, string format,
 int row_begin, int row_end,
 int col_begin, int col_end,
 int layer_begin, int layer_end
)
{
  //Creates the directory where the images will be stored.
  char buff_mkdir[2048];
  sprintf(buff_mkdir, "mkdir %s/MIP", directory.c_str());
  int err = system(buff_mkdir);

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];

  sprintf(buff, image_format, row_begin, col_begin, layer_begin);

  for(int nRow = row_begin; nRow <= row_end; nRow++){
    for(int nCol = col_begin; nCol <= col_end; nCol++){
      sprintf(buff, image_format, nRow, nCol, layer_begin);
      IplImage* mip = cvLoadImage(buff,0);
      for(int z = layer_begin + 1; z <= layer_end; z++){
        sprintf(buff, image_format, nRow, nCol, z);
        printf("Loading %s\n", buff);
        IplImage* toMip = cvLoadImage(buff,0);
        for(int x = 0; x < mip->width; x++){
          for(int y = 0; y < mip->height; y++){
            if( ((uchar *)(toMip->imageData + y*toMip->widthStep))[x] <
               ((uchar *)(mip->imageData   + y*mip->widthStep))[x] )
              ((uchar *)(mip->imageData   + y*mip->widthStep))[x] = 
                ((uchar *)(toMip->imageData + y*toMip->widthStep))[x];
          }
        }
        cvReleaseImage(&toMip);
      }
      sprintf(buff, "%s/MIP/%02i_%02i.jpg", directory.c_str(), nRow, nCol);
      cvSaveImage(buff, mip);
    }
  }
}

template <class T, class U>
void Cube<T,U>::save_as_image_stack(string dirname)
{

  float min_value = 1e6;
  float max_value = -1e6;

  if(sizeof(T)==4){
    for(int z = 0; z < cubeDepth; z++)
      for(int y = 0; y < cubeHeight; y++)
        for(int x = 0; x < cubeWidth; x++){
          if(min_value > at(x,y,z))
            min_value = at(x,y,z);
          if(max_value < at(x,y,z))
            max_value = at(x,y,z);
        }
  }

  printf("Cube<T,U>::save_as_image_stack saving images in %s [", dirname.c_str());
  char image_name[1024];
  if(sizeof(T)==4){
    for(int z = 0; z < cubeDepth; z++)
      {
        sprintf(image_name, "%s/%03i.png", dirname.c_str(), z);
        IplImage* toSave = cvCreateImage(cvSize(cubeWidth, cubeHeight), IPL_DEPTH_8U, 1);
        for(int y = 0; y < cubeHeight; y++)
          for(int x = 0; x < cubeWidth; x++)
            toSave->imageData[x + y*toSave->widthStep]
              = 255*(this->at(x,y,z)-min_value)/(max_value - min_value);
        cvSaveImage(image_name, toSave);
        printf("#"); fflush(stdout);
      }
  } else {
    for(int z = 0; z < cubeDepth; z++)
      {
        sprintf(image_name, "%s/%03i.png", dirname.c_str(), z);
        IplImage* toSave = cvCreateImage(cvSize(cubeWidth, cubeHeight), IPL_DEPTH_8U, 1);
        for(int y = 0; y < cubeHeight; y++)
          for(int x = 0; x < cubeWidth; x++)
            toSave->imageData[x + y*toSave->widthStep] = this->at(x,y,z);
        cvSaveImage(image_name, toSave);
        printf("#"); fflush(stdout);
      }
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::createMIPImage(string filename, bool minMax)
{
  if(filename == "")
    filename = "MIP.jpg";
  if(type == "uchar")
    {
      IplImage* output = cvCreateImage(cvSize(cubeWidth, cubeHeight), IPL_DEPTH_8U, 1);
      uint minimum_intensity;
      if(!minMax)
        minimum_intensity = 255;
      else
        minimum_intensity = 0;
      printf("creatingMIPImage [");
      for(int y = 0; y < cubeHeight; y++){
        for(int x = 0; x < cubeWidth; x++){
          if(!minMax)
            minimum_intensity = 255;
          else
            minimum_intensity = 0;
          for(int z = 0; z < cubeDepth; z++)
            {
              if( (this->at(x,y,z) < minimum_intensity) && !minMax)
                minimum_intensity = (uint)this->at(x,y,z);
              if( (this->at(x,y,z) > minimum_intensity) && minMax)
                minimum_intensity = (uint)this->at(x,y,z);
            }
          output->imageData[y*output->widthStep + x] = minimum_intensity;
        }
        if(y%100 == 0){
          printf("#");
          fflush(stdout);
        }
      }
      printf("]\n");
      cvSaveImage(filename.c_str(), output);
    }
  else{
    Image<float>* dm  = new Image<float>();
    dm->width = this->cubeWidth;
    dm->height = this->cubeHeight;
    printf("creatingMIPImage [");
    Image<float>* output = dm->create_blank_image_float(filename);
    output->put_all(0);
    float minimum_intensity = 1e8;
    if(!minMax)
      minimum_intensity = 1e8;
    else
      minimum_intensity = -1e8;

    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        if(!minMax)
          minimum_intensity = 1e8;
        else
          minimum_intensity = -1e8;
        for(int z = 0; z < cubeDepth; z++)
          {
            if((this->at(x,y,z)!=0) &&
               (this->at(x,y,z) < minimum_intensity) && 
               !minMax)
              {
                minimum_intensity = this->at(x,y,z);
                output->put(x,y,minimum_intensity);
              }
            if((this->at(x,y,z)!=0) &&
               (this->at(x,y,z) > minimum_intensity) && 
               minMax)
              {
                minimum_intensity = this->at(x,y,z);
                output->put(x,y,minimum_intensity);
              }
          }
      }
      if(y%100 == 0){
        printf("#");
        fflush(stdout);
      }
    }
    printf("]\n");
    output->save();
  }
}

template <class T, class U>
void Cube<T,U>::micrometersToIndexes(vector< float >& micrometers, vector< int >& indexes)
{
  indexes.push_back((int)(float(cubeWidth)/2 + micrometers[0]/voxelWidth));
  indexes.push_back((int)(float(cubeHeight)/2 - micrometers[1]/voxelHeight));
  indexes.push_back((int)(float(cubeDepth)/2 + micrometers[2]/voxelDepth));
}

template <class T, class U>
void Cube<T,U>::indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers)
{
  // micrometers[0] = (float)(-((int)cubeWidth)*voxelWidth/2   
                           // + indexes[0]*voxelWidth  + x_offset);
  // micrometers[1] = (float)( ((int)cubeHeight)*voxelHeight/2 
                            // - indexes[1]*voxelHeight - y_offset);
  // micrometers[2] = (float)(-((int)cubeDepth)*voxelDepth/2   
                           // + indexes[2]*voxelDepth  + z_offset);
  micrometers[0] = (float)(-((int)cubeWidth)*voxelWidth/2   
                           + indexes[0]*voxelWidth);
  micrometers[1] = (float)( ((int)cubeHeight)*voxelHeight/2 
                            - indexes[1]*voxelHeight);
  micrometers[2] = (float)(-((int)cubeDepth)*voxelDepth/2   
                           + indexes[2]*voxelDepth);


}

template <class T, class U>
void Cube<T,U>::print_statistics(string filename)
{
//   printf("%f %f %f\n", voxels[0][0][0], voxels[112][511][511], voxels[30][400][40]);
  //Will find the mean and the variance and print it. Also the max and the min
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += voxels[z][y][x];
          if(voxels[z][y][x] > max)
            max = voxels[z][y][x];
          if(voxels[z][y][x] < min)
            min = voxels[z][y][x];
        }

  mean = mean / (cubeDepth*cubeHeight*cubeWidth);
  printf("Cube mean value is %06.015f, max = %06.015f, min = %06.015f\n", mean, max, min);
}


template <class T, class U>
void Cube<T,U>::histogram(string filename)
{

  printf("Cube<T,U>::histogram [");
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += voxels[z][y][x];
          if(voxels[z][y][x] > max)
            max = voxels[z][y][x];
          if(voxels[z][y][x] < min)
            min = voxels[z][y][x];
        }

  float range = max - min;

  vector< int > boxes(100);
  for(int i = 0; i < 100; i++)
    boxes[i] = 0;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        boxes[(int)(floor(100*(this->at(x,y,z)-min)/range))] += 1;
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

  if(filename == ""){
    for(int i =0; i < boxes.size(); i++)
      printf("[%f %f] - %i\n", i*range/100, (i+1)*range/100,  boxes[i]);
    printf("\n");
  }
  else{
    std::ofstream out(filename.c_str());
    out << min << std::endl;
    out << max << std::endl;
    for(int i = 0; i < boxes.size(); i++)
      out << boxes[i] << std::endl;
    out.close();
  }
}



template <class T, class U>
void Cube<T,U>::histogram_ignoring_zeros(string filename)
{
  printf("Cube<T,U>::histogram [");
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += voxels[z][y][x];
          if(voxels[z][y][x] > max)
            max = voxels[z][y][x];
          if(voxels[z][y][x] < min)
            min = voxels[z][y][x];
        }

  float range = max - min;

  vector< int > boxes(100);
  for(int i = 0; i < 100; i++)
    boxes[i] = 0;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        if(this->at(x,y,z) == 0.0)
          continue;
        boxes[floor(100*(this->at(x,y,z)-min)/range)] += 1;
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

  if(filename == ""){
    for(int i =0; i < boxes.size(); i++)
      printf("%i ", boxes[i]);
    printf("\n");
  }
  else{
    std::ofstream out(filename.c_str());
    out << min << std::endl;
    out << max << std::endl;
    for(int i = 0; i < boxes.size(); i++)
      out << boxes[i] << std::endl;
    out.close();
  }
}

template <class T, class U>
void Cube<T,U>::apply_mask(string mask_nfo, string mask_vl, string output_nfo, string output_vl)
{
  Cube<uchar,ulong>* mask = new Cube<uchar, ulong>(mask_nfo, mask_vl);
  Cube<T,U>*     output = new Cube<T,U>(output_nfo, output_vl);

  printf("Cube<T,U>::apply_mask %s %s %s %s\n[", mask_nfo.c_str(), mask_vl.c_str(), output_nfo.c_str(), output_vl.c_str());
  for(int z = 0; z < mask->cubeDepth; z++){
    for(int y = 0; y < mask->cubeHeight; y++){
      for(int x = 0; x < mask->cubeWidth; x++){
        if(mask->at(x,y,z) == 255)
          output->put(x,y,z,this->at(x,y,z));
        else
          output->put(x,y,z,0);
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  delete mask;
  delete output;
}

/** The dimensions are on the voxels of the cube*/
template <class T, class U>
double Cube<T,U>::integral_between(int x0, int y0, int z0, int x1, int y1, int z1)
{
  // First step is to calculate in which dimension we have the greatest distance.

  double value_to_return = 0;

  //Case it is in the x
  if ( (abs(x1-x0) >= abs(y1-y0)) &&
       (abs(x1-x0) >= abs(z1-z0)) )
  {
    float my = ((float)(y1-y0))/(x1-x0);
    float mz = ((float)(z1-z0))/(x1-x0);
    float y = 0;
    float z = 0;
    if(min(x0,x1) == x0)
      {
        y = y0;
        z = z0;
      }
    if(min(x0,x1) == x1)
      {
        y = y1;
        z = z1;
      }
    for(int x = min(x0,x1); x <= max(x0,x1); x++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        value_to_return += this->at((int)roundf(x),(int)roundf(y),(int)roundf(z));
        y += my;
        z += mz;
      }
    value_to_return = value_to_return / (abs(x1-x0)+1);
  }

  //Case it is in the y
  if ( (abs(y1-y0) >= abs(x1-x0)) &&
       (abs(y1-y0) >= abs(z1-z0)) )
  {
    float mx = ((float)(x1-x0))/(y1-y0);
    float mz = ((float)(z1-z0))/(y1-y0);
    float x = 0;
    float z = 0;
    if(min(y0,y1) == y0)
      {
        x = x0;
        z = z0;
      }
    if(min(y0,y1) == y1)
      {
        x = x1;
        z = z1;
      }
    for(int y = min(y0,y1); y <= max(y0,y1); y++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        value_to_return += this->at((int)roundf(x),(int)roundf(y),(int)roundf(z));
        x += mx;
        z += mz;
      }
    value_to_return = value_to_return / (abs(y1-y0)+1);
  }

  //Case it is in the z
  if ( (abs(z1-z0) >= abs(y1-y0)) &&
       (abs(z1-z0) >= abs(x1-x0)) )
  {
    float my = ((float)(y1-y0))/(z1-z0);
    float mx = ((float)(x1-x0))/(z1-z0);
    float y = 0;
    float x = 0;
    if(min(z0,z1) == z0)
      {
        y = y0;
        x = x0;
      }
    if(min(z0,z1) == z1)
      {
        y = y1;
        x = x1;
      }

    for(int z = min(z0,z1); z <= max(z0,z1); z++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        value_to_return += this->at((int)roundf(x),(int)roundf(y),(int)roundf(z));
        y += my;
        x += mx;
      }
    value_to_return = value_to_return / (abs(z1-z0)+1);
  }

  return value_to_return;

}


template <class T, class U>
void Cube<T,U>::put_value_in_line(T value, int x0, int y0, int z0, int x1, int y1, int z1)
{
  // First step is to calculate in which dimension we have the greatest distance.

  double value_to_return = 0;
  int xt, yt, zt;

  //Case it is in the x
  if ( (abs(x1-x0) >= abs(y1-y0)) &&
       (abs(x1-x0) >= abs(z1-z0)) )
  {
    float my = ((float)(y1-y0))/(x1-x0);
    float mz = ((float)(z1-z0))/(x1-x0);
    float y = 0;
    float z = 0;
    if(min(x0,x1) == x0)
      {
        y = y0;
        z = z0;
      }
    if(min(x0,x1) == x1)
      {
        y = y1;
        z = z1;
      }
    for(int x = min(x0,x1); x <= max(x0,x1); x++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        if(x < 0) xt = 0; else if(x >= cubeWidth) xt = cubeWidth-1; else xt=x;
        if(y < 0) yt = 0; else if(y >= cubeHeight) yt = cubeHeight-1; else yt = y;
        if(z < 0) zt = 0; else if(z >= cubeDepth) zt = cubeDepth-1;   else zt = z;
        this->put((int)roundf(xt),(int)roundf(yt),(int)roundf(zt), value);
        y += my;
        z += mz;
      }
    value_to_return = value_to_return / (abs(x1-x0)+1);
  }

  //Case it is in the y
  if ( (abs(y1-y0) >= abs(x1-x0)) &&
       (abs(y1-y0) >= abs(z1-z0)) )
  {
    float mx = ((float)(x1-x0))/(y1-y0);
    float mz = ((float)(z1-z0))/(y1-y0);
    float x = 0;
    float z = 0;
    if(min(y0,y1) == y0)
      {
        x = x0;
        z = z0;
      }
    if(min(y0,y1) == y1)
      {
        x = x1;
        z = z1;
      }
    for(int y = min(y0,y1); y <= max(y0,y1); y++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        if(x < 0) xt = 0; else if(x >= cubeWidth) xt = cubeWidth-1; else xt=x;
        if(y < 0) yt = 0; else if(y >= cubeHeight) yt = cubeHeight-1; else yt = y;
        if(z < 0) zt = 0; else if(z >= cubeDepth) zt = cubeDepth-1;   else zt = z;
        this->put((int)roundf(xt),(int)roundf(yt),(int)roundf(zt), value);
        x += mx;
        z += mz;
      }
    value_to_return = value_to_return / (abs(y1-y0)+1);
  }

  //Case it is in the z
  if ( (abs(z1-z0) >= abs(y1-y0)) &&
       (abs(z1-z0) >= abs(x1-x0)) )
  {
    float my = ((float)(y1-y0))/(z1-z0);
    float mx = ((float)(x1-x0))/(z1-z0);
    float y = 0;
    float x = 0;
    if(min(z0,z1) == z0)
      {
        y = y0;
        x = x0;
      }
    if(min(z0,z1) == z1)
      {
        y = y1;
        x = x1;
      }

    for(int z = min(z0,z1); z <= max(z0,z1); z++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        if(x < 0) xt = 0; else if(x >= cubeWidth) xt = cubeWidth-1; else xt=x;
        if(y < 0) yt = 0; else if(y >= cubeHeight) yt = cubeHeight-1; else yt = y;
        if(z < 0) zt = 0; else if(z >= cubeDepth) zt = cubeDepth-1;   else zt = z;
        this->put((int)roundf(xt),(int)roundf(yt),(int)roundf(zt), value);
        y += my;
        x += mx;
      }
    value_to_return = value_to_return / (abs(z1-z0)+1);
  }
}


template <class T, class U>
void Cube<T,U>::put_value_in_ellipsoid
(T value, int x0, int y0, int z0, int rx, int ry, int rz)
{
  double dist;
  for(int z = max(0, z0-rz); z <= min((int)cubeDepth-1, z0+rz); z++)
    for(int y = max(0, y0-ry); y <= min((int)cubeHeight-1, y0+ry); y++)
      for(int x = max(0, x0-rx); x <= min((int)cubeWidth-1, x0+rx); x++){
        dist =  double((z-z0)*(z-z0))/(rz*rz);
        dist += double((y-y0)*(y-y0))/(ry*ry);
        dist += double((x-x0)*(x-x0))/(rx*rx);
        if(dist <= 1)
          put(x,y,z,value);
      }
}



template <class T, class U>
void Cube<T,U>::print()
{
  printf("Cube parameters:\n");
  printf("  indexes: %llu %llu %llu\n", cubeWidth, cubeHeight, cubeDepth);
  printf("  voxels : %f %f %f\n", voxelWidth, voxelHeight, voxelDepth);
}

template <class T, class U>
T Cube<T,U>::at(int x, int y, int z) {return voxels[z][y][x];}

template <class T, class U>
void Cube<T,U>::put(int x, int y, int z, T value) {voxels[z][y][x] = value;}

template <class T, class U>
void Cube<T,U>::put_all(T value) {
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        voxels[z][y][x] = value;
}


template <class T, class U>
U Cube<T,U>::integral_volume_at(int x, int y, int z) {return voxels_integral[z][y][x];}

template <class T, class U>
void Cube<T,U>::cut_cube(int x0, int y0, int z0, int x1, int y1, int z1, string name)
{
  Cube<T,U>* output = new Cube<T,U>();
  output->voxelWidth  = voxelWidth;
  output->voxelHeight = voxelHeight;
  output->voxelDepth  = voxelDepth;
  output->cubeWidth   = x1-x0;
  output->cubeHeight  = y1-y0;
  output->cubeDepth   = z1-z0;
  output->x_offset    = float(x1+x0)*this->voxelWidth /2 + x_offset;
  output->y_offset    = cubeHeight*voxelHeight/2 -
    float(y1+y0)*this->voxelHeight/2 + y_offset;
  output->z_offset    = float(z1+z0)*this->voxelDepth /2 + z_offset;
  string name_no_dir = name.substr(name.find_last_of("/\\")+1);
  output->filenameVoxelData = name_no_dir + ".vl";
  output->type = type;

  output->save_parameters(name + ".nfo");
  output->create_volume_file(name + ".vl");
  output->load_volume_data(name + ".vl");

  for(int z = z0; z< z1; z++)
    for(int y = y0; y < y1; y++)
      for(int x = x0; x < x1; x++)
        output->put(x-x0,y-y0,z-z0, this->at(x,y,z));

  delete output;
}

template <class T, class U>
Cube<T,U>*  Cube<T,U>::duplicate_clean(string name)
{
  string vl = ".vl";
  string nfo = ".nfo";

  Cube<T,U>* toReturn = new Cube<T,U>();
  toReturn->cubeHeight = cubeHeight;
  toReturn->cubeDepth  = cubeDepth;
  toReturn->cubeWidth  = cubeWidth;
  toReturn->voxelHeight = voxelHeight;
  toReturn->voxelDepth  = voxelDepth;
  toReturn->voxelWidth  = voxelWidth;
  toReturn->x_offset = x_offset;
  toReturn->y_offset = y_offset;
  toReturn->z_offset = z_offset;
  toReturn->directory = directory;
  if(sizeof(T)==1)
    toReturn->type = "uchar";
  else
    toReturn->type = "float";
  toReturn->filenameVoxelData = name + vl;
  toReturn->save_parameters(this->directory + name + nfo);
  toReturn->create_volume_file(this->directory + name + vl);
  toReturn->load_volume_data(this->directory + name + vl);

  return toReturn;
}



template <class T, class U>
Cube<float,double>*  Cube<T,U>::create_blank_cube(string name)
{
  string vl = ".vl";
  string nfo = ".nfo";

  if(fileExists(this->directory + name + nfo) &&
     fileExists(this->directory + name + vl))
    return new Cube<float, double>(this->directory + name + nfo);

  Cube<float,double>* toReturn = new Cube<float,double>();
  toReturn->cubeHeight = cubeHeight;
  toReturn->cubeDepth  = cubeDepth;
  toReturn->cubeWidth  = cubeWidth;
  toReturn->voxelHeight = voxelHeight;
  toReturn->voxelDepth  = voxelDepth;
  toReturn->voxelWidth  = voxelWidth;
  toReturn->x_offset = x_offset;
  toReturn->y_offset = y_offset;
  toReturn->z_offset = z_offset;
  toReturn->directory = directory;
  toReturn->type = "float";
  toReturn->filenameVoxelData = name + vl;
  toReturn->save_parameters(this->directory + name + nfo);
  toReturn->create_volume_file(this->directory + name + vl);
  toReturn->load_volume_data(this->directory + name + vl);

  return toReturn;
}

template <class T, class U>
Cube<uchar,ulong>* Cube<T,U>::create_blank_cube_uchar(string name)
{
  string vl = ".vl";
  string nfo = ".nfo";

  Cube<uchar,ulong>* toReturn = new Cube<uchar,ulong>();
  toReturn->cubeHeight = cubeHeight;
  toReturn->cubeDepth  = cubeDepth;
  toReturn->cubeWidth  = cubeWidth;
  toReturn->voxelHeight = voxelHeight;
  toReturn->voxelDepth  = voxelDepth;
  toReturn->voxelWidth  = voxelWidth;
  toReturn->x_offset = x_offset;
  toReturn->y_offset = y_offset;
  toReturn->z_offset = z_offset;
  toReturn->directory = directory;
  toReturn->type = "uchar";
  toReturn->filenameVoxelData = name + vl;
  toReturn->save_parameters(this->directory + name + nfo);
  toReturn->create_volume_file(this->directory + name + vl);
  toReturn->load_volume_data(this->directory + name + vl);

  return toReturn;

}

template <class T, class U>
void Cube<T,U>::create_gaussian_pyramid()
{
  //It will subsample the cube in /2 /2 /2 until it reaches the minimum size of the cube: 512, 512, 512.

  string name = this->filenameVoxelData.substr(0,filenameVoxelData.find_last_of("."));

  vector< float > Mask0;
  vector< float > Mask1;
  gaussian_mask(1.0,Mask0,Mask1);

  int orig_cubeWidth = cubeWidth;
  int orig_cubeHeight= cubeHeight;
  int orig_cubeDepth= cubeDepth;

  float orig_voxelWidth = voxelWidth;
  float orig_voxelHeight  = voxelHeight;
  float orig_voxelDepth   = voxelDepth;

  Cube<uchar, ulong>* previous = this;

//   int nIteration = 1;
//   while(  ((int)cubeWidth > 512) &&
//           ((int)cubeHeight > 512) &&
//           ((int)cubeDepth > 2)
//           )
  int nIteration = 2;
  for(nIteration = 2; nIteration <= 16; nIteration*=2)
    {
      printf("create_gaussian_pyramid: scale  %i\n", nIteration);

      Cube<float, double>* gaussian_blured = create_blank_cube("tmp");
      Cube<float, double>* tmp = create_blank_cube("tmp_tmp");
      previous->blur(1.0, gaussian_blured, tmp);

      printf("[");

      cubeWidth = cubeWidth/2;
      cubeHeight = cubeHeight/2;
      cubeDepth = cubeDepth/2;
      voxelDepth = voxelDepth*2;
      voxelHeight = voxelHeight*2;
      voxelWidth = voxelWidth*2;
//       nIteration*=2;

      char buff[1024];
      sprintf(buff, "%s_%i",name.c_str(),  nIteration);
      Cube<uchar, ulong>* out;
      out = create_blank_cube_uchar(buff);
      float integral = 0;
      int points_in_subsample = 0;
      for(int z = 0; z < out->cubeDepth; z++){
        for(int y = 0; y < out->cubeHeight; y++){
          for(int x = 0; x < out->cubeWidth; x++){
            integral = 0;
            points_in_subsample = 0;
            for(int z0 = 0; z0 < 2; z0++){
              for(int y0 = 0; y0<2; y0++){
                for(int x0 = 0; x0 < 2; x0++){
                  if( (x+x0 >= orig_cubeWidth) ||
                      (y+y0 >= orig_cubeHeight) ||
                      (z+z0 >= orig_cubeDepth))
                    continue;
                  integral+= gaussian_blured->at(x*2+x0,y*2+y0,z*2+z0);
                  points_in_subsample++;
                }
              }
            }
            out->put(x,y,z,(uchar)(integral/points_in_subsample));
            //End of the subsampling
          }
        }
        printf("#"); fflush(stdout);
      }
      previous = out;
      printf("]\n");

      //Removes the intermediate cubes
      buff[1024];
      sprintf(buff, "rm %s/tmp.nfo %s/tmp.vl %s/tmp_tmp.nfo %s/tmp_tmp.vl",
              directory.c_str(),  directory.c_str(),  directory.c_str(),
              directory.c_str());
      system(buff);
    }
  cubeWidth = orig_cubeWidth;
  cubeHeight = orig_cubeHeight;
  cubeDepth = orig_cubeDepth;
  voxelWidth = orig_voxelWidth; 
  voxelHeight = orig_voxelHeight;
  voxelDepth = orig_voxelDepth;
}

template <class T, class U>
void Cube<T,U>::create_gaussian_pyramid_2D()
{
  //It will subsample the cube in /2 /2 /2 until it reaches the minimum size of the cube: 512, 512, 512.

  vector< float > Mask0;
  vector< float > Mask1;
  gaussian_mask(1.0,Mask0,Mask1);

  int orig_cubeWidth = cubeWidth;
  int orig_cubeHeight= cubeHeight;
  int orig_cubeDepth= cubeDepth;

  float orig_voxelWidth = voxelWidth;
  float orig_voxelHeight  = voxelHeight;
  float orig_voxelDepth   = voxelDepth;

  Cube<uchar, ulong>* previous = this;

  string name = this->filenameVoxelData.substr(0,filenameVoxelData.find_last_of("."));

//   int nIteration = 1;
//   while(  ((int)cubeWidth > 512) &&
//           ((int)cubeHeight > 512) &&
//           ((int)cubeDepth > 2)
//           )
  char buff[1024];
  int nIteration = 2;
  for(nIteration = 2; nIteration <= 16; nIteration*=2)
    {
      printf("create_gaussian_pyramid: scale  %i\n", nIteration);
      printf("[");

      Cube<uchar, ulong>* out;
//       cubeDepth = cubeDepth/2;
//       voxelDepth = voxelDepth*2;

      if(voxelDepth > 2*voxelWidth){
        Cube<float, double>* gaussian_blured = create_blank_cube("tmp");
        Cube<float, double>* tmp = create_blank_cube("tmp_tmp");
        previous->blur_2D(1.0, gaussian_blured, tmp);

        cubeWidth = cubeWidth/2;
        cubeHeight = cubeHeight/2;
        voxelHeight = voxelHeight*2;
        voxelWidth = voxelWidth*2;

        char buff[1024];
        sprintf(buff, "%s_%i",name.c_str(),  nIteration);
        out = create_blank_cube_uchar(buff);
        float integral = 0;
        int points_in_subsample = 0;
        for(int z = 0; z < out->cubeDepth; z++){
          for(int y = 0; y < out->cubeHeight; y++){
            for(int x = 0; x < out->cubeWidth; x++){
              integral = 0;
              points_in_subsample = 0;
              //             for(int z0 = 0; z0 < 2; z0++){
              for(int y0 = 0; y0<2; y0++){
                for(int x0 = 0; x0 < 2; x0++){
                  if( (x+x0 >= orig_cubeWidth) ||
                      (y+y0 >= orig_cubeHeight) )
                    continue;
                  integral+= gaussian_blured->at(x*2+x0,y*2+y0,z);
                  points_in_subsample++;
                }
                //               }
              }
              out->put(x,y,z,(uchar)(integral/points_in_subsample));
              //End of the subsampling
            }
          }
          printf("#"); fflush(stdout);
        }
      } else {
        Cube<float, double>* gaussian_blured = create_blank_cube("tmp");
        Cube<float, double>* tmp = create_blank_cube("tmp_tmp");
        previous->blur(1.0, gaussian_blured, tmp);

        cubeDepth = cubeDepth/2;
        voxelDepth = voxelDepth*2;
        cubeWidth = cubeWidth/2;
        cubeHeight = cubeHeight/2;
        voxelHeight = voxelHeight*2;
        voxelWidth = voxelWidth*2;

        char buff[1024];
        sprintf(buff, "%s_%i",name.c_str(),  nIteration);
        out = create_blank_cube_uchar(buff);
        float integral = 0;
        int points_in_subsample = 0;
        for(int z = 0; z < out->cubeDepth; z++){
          for(int y = 0; y < out->cubeHeight; y++){
            for(int x = 0; x < out->cubeWidth; x++){
              integral = 0;
              points_in_subsample = 0;
              for(int z0 = 0; z0 < 2; z0++){
                for(int y0 = 0; y0<2; y0++){
                  for(int x0 = 0; x0 < 2; x0++){
                    if( (x+x0 >= orig_cubeWidth) ||
                        (y+y0 >= orig_cubeHeight) ||
                        (z+z0 >= orig_cubeDepth))
                      continue;
                    integral+= gaussian_blured->at(x*2+x0,y*2+y0,z*2+z0);
                    points_in_subsample++;
                  }
                }
              }
              out->put(x,y,z,(uchar)(integral/points_in_subsample));
              //End of the subsampling
            }
          }
          printf("#"); fflush(stdout);
        }
        previous = out;
        printf("]\n");

      }



      previous = out;
      printf("]\n");

      //Removes the intermediate cubes
      buff[1024];
      sprintf(buff, "rm %s/tmp.nfo %s/tmp.vl %s/tmp_tmp.nfo %s/tmp_tmp.vl",
              directory.c_str(),  directory.c_str(),  directory.c_str(),
              directory.c_str());
      system(buff);
    }
  cubeWidth = orig_cubeWidth;
  cubeHeight = orig_cubeHeight;
  cubeDepth = orig_cubeDepth;
  voxelWidth = orig_voxelWidth; 
  voxelHeight = orig_voxelHeight;
  voxelDepth = orig_voxelDepth;
}



template <class T, class U>
void Cube<T,U>::apply_affine_transform
(gsl_matrix* transform,
 IplImage*   orig,
 IplImage*   dest,
 vector< vector< int > >* mask
 )
{
  gsl_vector* c_o = gsl_vector_alloc(3);
  gsl_vector* c_d = gsl_vector_alloc(3);
  //Hack to do not include the last column of the image
  for(int x = 0; x < orig->width-2; x++){
    for(int y = 0; y < orig->height; y++){
      gsl_vector_set(c_o,0,x);
      gsl_vector_set(c_o,1,y);
      gsl_vector_set(c_o,2,1);
      gsl_blas_dgemv(CblasNoTrans, 1.0, transform, c_o, 0, c_d);
      if( (gsl_vector_get(c_d,0) < 0) ||
          (gsl_vector_get(c_d,0) >= dest->width) ||
          (gsl_vector_get(c_d,1) < 0) ||
          (gsl_vector_get(c_d,1) >= dest->height)
          )
        continue;
      ((uchar *)(dest->imageData + (int)gsl_vector_get(c_d,1)*dest->widthStep))
        [(int)gsl_vector_get(c_d,0)] =
         ((uchar *)(orig->imageData + y*orig->widthStep))[x];
      if(mask!=NULL){
        (*mask)[(int)gsl_vector_get(c_d,0)][(int)gsl_vector_get(c_d,1)] = 1;
      }
    }
  }
  gsl_vector_free(c_o);
  gsl_vector_free(c_d);
}


template <class T, class U>
double Cube<T,U>::cross_correlate
( gsl_matrix* trns_init,
  IplImage* stitched,
  vector<vector< int > >& mask,
  IplImage* img,
  int ey,
  int ex,
  WhereToCorrelate where
  )
{
  gsl_matrix* trns_curr  = gsl_matrix_alloc(3,3);
  gsl_matrix* trns_tmp   = gsl_matrix_alloc(3,3);
  gsl_vector* c_d = gsl_vector_alloc(3);
  gsl_vector* c_o = gsl_vector_alloc(3);
  gsl_vector_set(c_d,2,1.0);
  gsl_vector_set(c_o,2,1.0);

  vector<double> img1;
  vector<double> img2;

  img1.resize(0);
  img2.resize(0);
  gsl_matrix_set_identity(trns_tmp);
  gsl_matrix_set(trns_tmp, 0, 2, ex);
  gsl_matrix_set(trns_tmp, 1, 2, ey);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
                 trns_init, trns_tmp, 0.0,  trns_curr);

  if(where == STITCH_UP){
    for(int y = 0; y < -ey; y++){
      for(int x = 0; x < img->width; x++){
        gsl_vector_set(c_o,0,x);
        gsl_vector_set(c_o,1,y);
        gsl_blas_dgemv(CblasNoTrans, 1.0, trns_curr, c_o, 0, c_d);
        if( (gsl_vector_get(c_d,0) >= 0) &&
            (gsl_vector_get(c_d,0) < stitched->width) &&
            (gsl_vector_get(c_d,1) >= 0) &&
            (gsl_vector_get(c_d,1) < stitched->height) &&
            (mask[(int)gsl_vector_get(c_d,0)][(int)gsl_vector_get(c_d,1)] == 1)
            )
          {
            img1.push_back(((uchar *)(stitched->imageData +
                                      (int)gsl_vector_get(c_d,1)*stitched->widthStep))
                           [(int)gsl_vector_get(c_d,0)]);
            img2.push_back(((uchar *)(img->imageData + y*img->widthStep))[x]);
          }
      } // x loop
    } //y loop
  }

  if(where == STITCH_LEFT){
    //This is to prevent the left border
    for(int x = 0; x < -ex; x++){
        for(int y = 0; y < img->height; y++){
          gsl_vector_set(c_o,0,x);
          gsl_vector_set(c_o,1,y);
          gsl_blas_dgemv(CblasNoTrans, 1.0, trns_curr, c_o, 0, c_d);
          if( (gsl_vector_get(c_d,0) >= 0) &&
              (gsl_vector_get(c_d,0) < stitched->width) &&
              (gsl_vector_get(c_d,1) >= 0) &&
              (gsl_vector_get(c_d,1) < stitched->height) &&
              (mask[(int)gsl_vector_get(c_d,0)][(int)gsl_vector_get(c_d,1)] == 1)
              )
            {
              img1.push_back(((uchar *)(stitched->imageData +
                                        (int)gsl_vector_get(c_d,1)*stitched->widthStep))
                             [(int)gsl_vector_get(c_d,0)]);
              img2.push_back(((uchar *)(img->imageData + y*img->widthStep))[x]);
            }
        } // x loop
      } //y loop
  }

  //In this case the correlation is done on up and left
  if(where == STITCH_LEFTUP){
    //This is to prevent the left border
    for(int x = 0; x < -ex; x++){
        for(int y = 0; y < img->height; y++){
          gsl_vector_set(c_o,0,x);
          gsl_vector_set(c_o,1,y);
          gsl_blas_dgemv(CblasNoTrans, 1.0, trns_curr, c_o, 0, c_d);
          if( (gsl_vector_get(c_d,0) >= 0) &&
              (gsl_vector_get(c_d,0) < stitched->width) &&
              (gsl_vector_get(c_d,1) >= 0) &&
              (gsl_vector_get(c_d,1) < stitched->height) &&
              (mask[(int)gsl_vector_get(c_d,0)][(int)gsl_vector_get(c_d,1)] == 1)
              )
            {
              img1.push_back(((uchar *)(stitched->imageData +
                                        (int)gsl_vector_get(c_d,1)*stitched->widthStep))
                             [(int)gsl_vector_get(c_d,0)]);
              img2.push_back(((uchar *)(img->imageData + y*img->widthStep))[x]);
            }
        } // x loop
    } //y loop
    for(int y = 0; y < -ey; y++){
      for(int x = 0; x < img->width; x++){
        gsl_vector_set(c_o,0,x);
        gsl_vector_set(c_o,1,y);
        gsl_blas_dgemv(CblasNoTrans, 1.0, trns_curr, c_o, 0, c_d);
        if( (gsl_vector_get(c_d,0) >= 0) &&
            (gsl_vector_get(c_d,0) < stitched->width) &&
            (gsl_vector_get(c_d,1) >= 0) &&
            (gsl_vector_get(c_d,1) < stitched->height) &&
            (mask[(int)gsl_vector_get(c_d,0)][(int)gsl_vector_get(c_d,1)] == 1)
            )
          {
            img1.push_back(((uchar *)(stitched->imageData +
                                      (int)gsl_vector_get(c_d,1)*stitched->widthStep))
                           [(int)gsl_vector_get(c_d,0)]);
            img2.push_back(((uchar *)(img->imageData + y*img->widthStep))[x]);
          }
      } // x loop
    } //y loop
  }
  //In the case there is no overlap between the images
  if(img1.size()==0)
    return -1;

  gsl_vector* s1 = gsl_vector_alloc(img1.size());
  gsl_vector* s2 = gsl_vector_alloc(img2.size());
  double m_s1 = 0;
  double m_s2 = 0;
  for(int i = 0; i < img1.size(); i++){
    gsl_vector_set(s1, i, img1[i]);
    gsl_vector_set(s2, i, img2[i]);
    m_s1 += img1[i];
    m_s2 += img2[i];
  }
  m_s1 = m_s1/img1.size();
  m_s2 = m_s2/img2.size();
  gsl_vector_add_constant(s1, -m_s1);
  gsl_vector_add_constant(s2, -m_s2);
  double sigma_1 = 0;
  double sigma_2 = 0;
  double cross_prod = 0;
  gsl_blas_ddot(s1,s1,&sigma_1);
  gsl_blas_ddot(s2,s2,&sigma_2);
  gsl_blas_ddot(s1,s2,&cross_prod);
  double n_cc = cross_prod/sqrt(sigma_1*sigma_2);

  gsl_vector_free(s1);
  gsl_vector_free(s2);

  gsl_matrix_free(trns_curr);
  gsl_matrix_free(trns_tmp);
  gsl_vector_free(c_d);
  gsl_vector_free(c_o);

  return n_cc;
}

template <class T, class U>
gsl_matrix* Cube<T,U>::find_affine_transform_from_correlation
( gsl_matrix* trns_init,
  IplImage* stitched,
  vector<vector< int > >& mask,
  IplImage* img,
  WhereToCorrelate where
  )
{

  gsl_matrix* trns_final = gsl_matrix_alloc(3,3);

  double n_cc_max = -1e6;

  int ex_f = 0;
  int ey_f = 0;

  double n_cc;

  if(where == STITCH_UP){
    for(int ey = -1; ey >= -15; ey--){
      for(int ex = -15; ex <= 15; ex++){
        n_cc = cross_correlate(trns_init, stitched, mask,
                               img, ey, ex, STITCH_UP);
        if(n_cc > n_cc_max){
          n_cc_max = n_cc;
          ex_f = ex;
          ey_f = ey;
        }
      }
    }
  }

  if(where == STITCH_LEFT){
    for(int ex = -1; ex >= -15; ex--){
      for(int ey = -15; ey <= 15; ey++){
        n_cc = cross_correlate(trns_init, stitched, mask,
                               img, ey, ex, STITCH_LEFT);
//         printf("%i %i %f\n", ex, ey, n_cc);
        if(n_cc > n_cc_max){
          n_cc_max = n_cc;
          ex_f = ex;
          ey_f = ey;
        }
      }
    }
  }

  if(where == STITCH_LEFTUP){
    //Elliminates the border of the image (left and right)
    for(int ex = -1; ex >= -15; ex--){
      for(int ey = -1; ey >= -15; ey--){
        n_cc = cross_correlate(trns_init, stitched, mask,
                               img, ey, ex, STITCH_LEFTUP);
        if(n_cc > n_cc_max){
          n_cc_max = n_cc;
          ex_f = ex;
          ey_f = ey;
        }
      }
    }
  }


  printf("The ofset found is %i %i \n", ex_f, ey_f);

  gsl_matrix* trns_tmp   = gsl_matrix_alloc(3,3);
  gsl_matrix_set_identity(trns_tmp);
  gsl_matrix_set(trns_tmp,0,2,ex_f);
  gsl_matrix_set(trns_tmp,1,2,ey_f);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
                 trns_init, trns_tmp, 0.0,  trns_final);
  gsl_matrix_free(trns_tmp);

  return trns_final;

}


template <class T, class U>
void Cube<T,U>::create_cube_from_directory_matrix_with_affine_transformation
(
 string directory, string format,
 int row_begin, int row_end,
 int col_begin, int col_end,
 int layer_begin, int layer_end,
 float voxelWidth, float voxelHeight,
 float voxelDepth, string name2
)
{
  create_directory_matrix_MIP(directory, format, row_begin, row_end,
                              col_begin, col_end, layer_begin, layer_end);

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];
  sprintf(buff, image_format, row_begin, col_begin, layer_begin);
  IplImage* pepe = cvLoadImage(buff,0);

  string parameters_file = directory + "/" + name2 + ".nfo";
  std::ofstream out_w(parameters_file.c_str());
  out_w << "cubeWidth " << pepe->width*(col_end - col_begin +1) << std::endl;
  out_w << "cubeHeight " << pepe->height*(row_end - row_begin +1) << std::endl;
  out_w << "cubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth << std::endl;
  out_w << "voxelHeight " << voxelHeight << std::endl;
  out_w << "voxelDepth " << voxelDepth << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w << "type  uchar" << std::endl;
  out_w << "cubeFile " << name2 << ".vl" << std::endl;
  out_w.close();

  this->cubeWidth   = pepe->width*(col_end - col_begin +1)     ;
  this->cubeHeight  = pepe->height*(row_end - row_begin +1)    ;
  this->cubeDepth   = layer_end - layer_begin + 1 ;
  this->voxelWidth  = voxelWidth    ;
  this->voxelHeight = voxelHeight   ;
  this->voxelDepth  = voxelDepth    ;


  string name = directory + "/" + name2 + ".vl";
  create_volume_file(name);
  load_volume_data(name);


  //Starts the stitching :) Up to now it was just preparation

  IplImage* img = cvLoadImage(buff, 0);
  int width = img->width;
  int height = img->height;
  IplImage* result = cvCreateImage(cvSize( (col_end - col_begin+1)*img->width,
                                           (row_end - row_begin+1)*img->height),
                                   img->depth, img->nChannels);
  cvReleaseImage(&img);

  //Creates the mask.
  vector< vector< int > > mask;
  for(int i = 0; i < result->width; i++){
    vector<int> pepe(result->height);
    mask.push_back(pepe);
    for(int j = 0; j < result->height; j++){
      mask[i][j] = 0;
    }
  }

  //Creates the registration matrix
  vector< vector< gsl_matrix* > > trns;
  for(int i = 0; i < row_end - row_begin + 1; i++){
    vector< gsl_matrix* > pp(col_end - col_begin + 1);
    trns.push_back(pp);
  }
  for(int i = 0; i < row_end - row_begin + 1; i++)
    for(int j = 0; j < col_end - col_begin + 1; j++)
      trns[i][j] = gsl_matrix_alloc(3,3);

  gsl_matrix_set_identity(trns[0][0]);
  gsl_matrix* trans_to_next = gsl_matrix_alloc(3,3);
  gsl_matrix_set_identity(trans_to_next);

  //Starts the stitching
  for(int nRow = row_begin; nRow <= row_end; nRow++){
    if( nRow > row_begin){
      gsl_matrix_set_identity(trans_to_next);
      gsl_matrix_set(trans_to_next, 1, 2, height);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
                     trns[nRow - row_begin-1][0],
                     trans_to_next, 0.0, trns[nRow - row_begin][0] );
    }

    for(int nCol = col_begin; nCol <= col_end; nCol++){
      sprintf(buff, "%s/MIP/%02i_%02i.jpg", directory.c_str(),nRow, nCol);
      printf("Image: %s\n", buff);
      IplImage* img = cvLoadImage(buff, 0);

      //Creating the approximation for the transformation matrix
      if( (nCol > col_begin) && (nRow > row_begin)){
        //Really needing to play here
        gsl_matrix_set_identity(trans_to_next);
        gsl_matrix_set(trans_to_next, 0, 2,
                       gsl_matrix_get(trns[nRow-row_begin][nCol-col_begin-1],0,2)+
                       img->width-2);
        gsl_matrix_set(trans_to_next, 1, 2,
                       gsl_matrix_get(trns[nRow-row_begin-1][nCol-col_begin],1,2)+
                       img->height
                       );
        gsl_matrix_memcpy(trns[nRow-row_begin][nCol-col_begin],trans_to_next);
      } else
        if(nCol > col_begin){
          gsl_matrix_set_identity(trans_to_next);
          gsl_matrix_set(trans_to_next, 0, 2, img->width-2);
          gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
                         trns[nRow - row_begin][nCol - col_begin -1],
                         trans_to_next, 0.0, trns[nRow - row_begin][nCol - col_begin] );
        }

      // Refines the transformation matrix
      if( (nCol == col_begin) && (nRow == row_begin)){
      }
      else if( (nRow == row_begin) ){
        trans_to_next = find_affine_transform_from_correlation
          ( trns[nRow-row_begin][nCol-col_begin], result, mask, img,  STITCH_LEFT);
        gsl_matrix_memcpy(trns[nRow-row_begin][nCol-col_begin], trans_to_next);
      }

      else if( (nCol == col_begin) ){
        trans_to_next = find_affine_transform_from_correlation
          ( trns[nRow-row_begin][nCol-col_begin], result, mask, img,  STITCH_UP);
        gsl_matrix_memcpy(trns[nRow-row_begin][nCol-col_begin], trans_to_next);
      }
      else{
        trans_to_next = find_affine_transform_from_correlation
          ( trns[nRow-row_begin][nCol-col_begin], result, mask, img,  STITCH_LEFTUP);
        gsl_matrix_memcpy(trns[nRow-row_begin][nCol-col_begin], trans_to_next);
      }

      //This is to create the MIP image
      apply_affine_transform( trns[nRow - row_begin][nCol - col_begin],
                              img, result, &mask);

      //Now it comes the actual cube creation
      gsl_vector* c_o = gsl_vector_alloc(3);
      gsl_vector* c_d = gsl_vector_alloc(3);

      //Puts the pixels in the cube
      for(int z = layer_begin; z <= layer_end; z++){
        sprintf(buff, image_format, nRow, nCol, z);
        IplImage* curr = cvLoadImage(buff,0);

        printf("%s\n",buff);

        for(int x = 0; x < curr->width; x++){
          for(int y = 0; y < curr->width; y++){
            gsl_vector_set(c_o,0,x);
            gsl_vector_set(c_o,1,y);
            gsl_vector_set(c_o,2,1);
            gsl_blas_dgemv(CblasNoTrans, 1.0,
                           trns[nRow - row_begin][nCol - col_begin],
                           c_o, 0, c_d);
            //Check the bounds
            if( (gsl_vector_get(c_d,0) < 0) ||
                (gsl_vector_get(c_d,0) >= result->width) ||
                (gsl_vector_get(c_d,1) < 0) ||
                (gsl_vector_get(c_d,1) >= result->height)
                )
              continue;

            this->put((int)gsl_vector_get(c_d,0),(int)gsl_vector_get(c_d,1),z-layer_begin,
                      ((uchar *)(curr->imageData + y*curr->widthStep))[x]);
          }
        }
        cvReleaseImage(&curr);
      }
      gsl_vector_free(c_o);
      gsl_vector_free(c_d);



      cvReleaseImage(&img);
    }
  }
  sprintf(buff, "%s/MIP/stitched_%s.jpg", directory.c_str(), name2.c_str());

  cvSaveImage(buff,result);
}


template <class T, class U>
void Cube<T,U>::create_cube_from_float_images
(
 string format,
 float begin,
 float end,
 float increment,
 float voxelWidth,
 float voxelHeight,
 float voxelDepth,
 string cubeName
 )
{
//   char buff[1024];
//   sprintf(buff, format.c_str(), begin);
//   Image<float>* img = new Image<float>(buff);
//   Cube<float,double>* to
//   printf("%i %i\n", img->width, img->height);



}


#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intensityFeature.h"
#include "Cloud.h"
#include "limits.h"

// TODO : Add sigmoid and polynomial kernels
enum eKernelType { KT_LINEAR, KT_SSD, KT_RBF};
eKernelType kernelType = KT_RBF;

#define SIGMA_SQUARE 0.015625f //0.125^2
//#define SIGMA_SQUARE 10

response_type getIntensityFeature(unsigned char *test_img,
                        int width, int height,
                        char* weak_learner_param,
                        xImage* img)
{
  const int patchDist = 8;
  int patchDiameter = 2*patchDist+1;
  int patchSize = patchDiameter * patchDiameter;

/*
  //printf("weak_learner_param %s\n",weak_learner_param);

  int i;
  //sscanf(weak_learner_param,"ax%day%dbx%dby%d",&col1,&row1,&col2,&row2);
  int cloudId = -1;
  if(sscanf(weak_learner_param,"IT_%d",&cloudId)==EOF)
    {
      printf("getIntensityFeature: Error while parsing the string\n");
      return -1;
    }
  if(cloudId < 0)
    {
      printf("getIntensityFeature: Error while parsing the string\n");
      return -1;
    }

  // Index starts at 0
  int fileId = (int) cloudId/nbPointsPerCloud;

  //printf("fileId %d\n",fileId);
  //printf("fileId %d\n",list_clouds.size());

  if(list_clouds.size()<=fileId)
    {
      printf("getIntensityFeature: list_clouds.size()<=fileId %d %d\n",list_clouds.size(),fileId);
      return -1;
    }

  //printf("fileId %d\n",fileId);
  
  //printf("f %d id %d\n", files.size(),fileId);

  // Open cloud file and read idCloud-th point
  int relCloudId = cloudId - fileId * nbPointsPerCloud;
  //Cloud* point_set = new Cloud(dir + files[fileId]);
  Cloud* point_set = list_clouds[fileId];

  //printf("point_set %x\n",point_set);
  //printf("point_set %x\n",point_set->points);

  //printf("%s %d %d\n", point_set->points.size(), relCloudId);
  
  if(point_set->points.size()<=relCloudId)
    {
      printf("getIntensityFeature: point_set->points.size()<=relCloudId %d %d\n",point_set->points.size(),relCloudId);
      return -1;
    }

  Point* pt = dynamic_cast<Point*>(point_set->points[relCloudId]);

  //printf("Pt %f %f\n", pt->coords[0], pt->coords[1]);

  //string img_dir = "/localhome/aurelien/Documents/EM/raw_mitochondria2/originals/";
  //string img_filename = img_dir + getNameFromPathWithoutExtension(files[fileId]);
  //img_filename += ".png";
  //IplImage* img = cvLoadImage(img_filename.c_str(),1);
  
  //xImage* img = list_images[fileId];
  xImage* img = list_images[0];

  if(img == 0)
    {
      printf("getIntensityFeature: Error while loading image number %d\n", fileId);
      return -1;
    }

  // micrometers to indexes
  int indexes[2];
  indexes[0] = (int)pt->coords[0];
  indexes[1] = (int)(img->height -0.001 - pt->coords[1]);

  //printf("Pt %d %d, %d %d depth : %d\n", indexes[0], indexes[1], patchDiameter, img->nChannels,img->depth);
*/

  response_type K = 0;
  unsigned char* codebook_patch = img->data;

  // Compute Kernel function      
  switch(kernelType)
    {
    case KT_RBF:
      {
        double diff;
        double ssd=0;
        double tempK;
        for(int i=0;i<patchSize;i++)
          {
            //K += test_img[i] - codebook_patch[i];
            //printf("%u %u\n", test_img[i], codebook_patch[i]);

            diff = (test_img[i] - codebook_patch[i])/255.0;
            ssd += (diff*diff);
          }
        //printf("%f ",ssd);
        // TODO : need to rescale K between 0 and 1
        //ofstream ofs("ITvalues.txt",ios::app);
        //ofs << tempK << endl;
        //ofs.write(K);
        //ofs.close();
        //const double minK = 238169;
        //const double maxK = 2.1806e+09;
        //tempK = (((double)ssd)-minK)/maxK;

        //printf("%f ",tempK);
        //K = exp(-(double)fabs(K)*SIGMA_SQUARE)*INT_MAX;
        K = exp(-(double)(ssd)*SIGMA_SQUARE)*INT_MAX;
        //printf("%d ",K);
      }
      break;
    case KT_SSD:
      {
        int temp;
        for(int i=0;i<patchSize;i++)
          {
            temp = test_img[i] - codebook_patch[i];
            //printf("%u %u\n", test_img[i], codebook_patch[i]);
            K += temp*temp;
          }
        K = sqrt(K);
      }
      break;
    case KT_LINEAR:
      // Compute dot product
      for(int i=0;i<patchSize;i++)
        {
          K += (test_img[i]*codebook_patch[i]);
        }
      // Square it
      K *= K;
      break;
    }

  return K;
}


/*
int getIntensityFeature2(unsigned char *test_img, int width, int height, char* weak_learner_param)
{
  const int patchDist = 8;
  int patchDiameter = 2*patchDist+1;
  int patchSize = patchDiameter * patchDiameter;

  //printf("weak_learner_param %s\n",weak_learner_param);

  int i;
  //sscanf(weak_learner_param,"ax%day%dbx%dby%d",&col1,&row1,&col2,&row2);
  int cloudId = -1;
  if(sscanf(weak_learner_param,"IT_%d",&cloudId)==EOF)
    {
      printf("getIntensityFeature: Error while parsing the string\n");
      return -1;
    }
  if(cloudId < 0)
    {
      printf("getIntensityFeature: Error while parsing the string\n");
      return -1;
    }

  const int nbPointsPerFile = 600;
  string dir("/localhome/aurelien/Sources/EM/svm_test/Model-8-6000-3-i/");
  vector<string> files;
  get_files_in_dir(dir,files,"cl");
  int fileId = (int) cloudId/nbPointsPerFile;

  if(files.size()<fileId)
    {
      printf("getIntensityFeature: Error while looking for cloud point\n");
      return -1;
    }

  //printf("f %d id %d\n", files.size(),fileId);

  // Open cloud file and read idCloud-th point
  int relCloudId = cloudId - fileId * nbPointsPerFile;
  Cloud* point_set = new Cloud(dir + files[fileId]);

  printf("%s %d %d\n", files[fileId].c_str(), point_set->points.size(), relCloudId);
  
  Point* pt = dynamic_cast<Point*>(point_set->points[relCloudId]);

  printf("Pt %f %f\n", pt->coords[0], pt->coords[1]);

  string img_dir = "/localhome/aurelien/Documents/EM/raw_mitochondria2/originals/";
  string img_filename = img_dir + getNameFromPathWithoutExtension(files[fileId]);
  img_filename += ".png";
  IplImage* img = cvLoadImage(img_filename.c_str(),1);

  if(img == 0)
    {
      printf("getIntensityFeature: Error while loading %s\n", img_filename.c_str());
      return -1;
    }

  // micrometers to indexes
  int indexes[2];
  indexes[0] = (int)pt->coords[0];
  indexes[1] = (int)(img->height -0.001 - pt->coords[1]);

  //printf("Pt %d %d, %d %d depth : %d\n", indexes[0], indexes[1], patchDiameter, img->nChannels,img->depth);

  int K = 0;
  if(indexes[0] >= patchDist && indexes[1] >= patchDist
     && indexes[0] < img->width-patchDist && indexes[1] < img->height-patchDist)
    {
      // Extract patch 1
      i=0;
      unsigned char codebook_patch[patchSize];
      for(int x=indexes[0]-patchDist;x<=indexes[0]+patchDist;x++)
        for(int y=indexes[1]-patchDist;y<=indexes[1]+patchDist;y++)
          {
            codebook_patch[i] = ((img->imageData + img->widthStep*y))[x*img->nChannels];            
            i++;
          }
      //delete[] img;
      //printf("\n");

      // Compute Kernel function      
#ifdef RBF
      int temp;
      for(i=0;i<patchSize;i++)
        {
          temp = test_img[i] - codebook_patch[i];
          //printf("%u %u\n", test_img[i], codebook_patch[i]);
          K += temp*temp;
        }
      K *= K;
#else
      // Linear kernel

      // Compute dot product
      for(i=0;i<patchSize;i++)
        {
          K += (test_img[i]*codebook_patch[i]);
        }
      // Square it
      K *= K;
#endif
      //delete[] point_set;
    }

  //cvReleaseImage(&img);
  return K;
}
*/

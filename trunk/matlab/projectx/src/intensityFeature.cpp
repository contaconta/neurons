#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "Cloud.h"

int getIntensityFeature(unsigned int *img, int width, int height, char* weak_learner_param)
{
  /*
  char temp[20];
  int size;
  char* token=strchr(weak_learner_param,'_');

  if(token == 0)
    return -1;

  size = weak_learner_param-token;
  strncpy(temp,weak_learner_param,size);
  temp[size] = 0;
  printf("x %s\n",temp);
  int x = atoi(temp);

  char* token2=strchr(token,'_');
  size = token2-token;
  strncpy(temp,token,size);
  temp[size] = 0;
  printf("y %s\n",temp);
  int y = atoi(temp);
  */

  const int patchDist = 10;
  const int patchSize = 2*patchSize+1;

  int i;
  //sscanf(weak_learner_param,"ax%day%dbx%dby%d",&col1,&row1,&col2,&row2);
  int cloudId = -1;
  int row1, col1;
  int row2 = -1;
  int col2 = -1;
  if(sscanf(weak_learner_param,"i%dx%dy%d",&cloudId,&col2,&row2)==EOF)
    {
      printf("getIntensityFeature: Error while parsing the string\n");
      return -1;
    }
  if(row2 < 0 || col2 < 0 || cloudId < 0)
    {
      printf("getIntensityFeature: Error while parsing the string\n");
      return -1;
    }

  const int nbPointsPerFile = 600;
  string dir("/localhome/aurelien/Sources/EM/svm_test/Model-8-6000-3-i");
  vector<string> files;
  get_files_in_dir(dir,files,".cl");
  int fileId = (int) cloudId/nbPointsPerFile;

  if(files.size()<fileId)
    {
      printf("getIntensityFeature: Error while looking for cloud point\n");
      return -1;
    }

  // Open cloud file and read idCloud-th point
  int relCloudId = cloudId - fileId * nbPointsPerFile;
  Cloud* point_set = new Cloud(files[fileId]);
  Point* pt = dynamic_cast<Point*>(point_set->points[relCloudId]);

  // Extract patch 1
  i=0;
  char patch1[patchSize];
  for(int x=col1-patchDist;x<=col1+patchDist;x++)
    for(int y=row1-patchDist;y<=row1+patchDist;y++)
      {
        patch1[i] = img[y*width+x];
        i++;
      }

  // Extract patch 2
  i=0;
  char patch2[patchSize];
  for(int x=col2-patchDist;x<=col2+patchDist;x++)
    for(int y=row2-patchDist;y<=row2+patchDist;y++)
      {
        patch2[i] = img[y*width+x];
        i++;
      }

  // Compute Kernel function
  int K = 0;
  int temp;
  for(i=0;i<patchSize;i++)
    {
      temp = patch2[i] - patch1[i];
      K += temp*temp;
    }
  K = (int)sqrt(K);

  return K;
}

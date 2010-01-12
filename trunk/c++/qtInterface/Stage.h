#ifndef STAGE_H_
#define STAGE_H_

#include <glew.h>
#include <QGLWidget>
#include "Axis.h"
#include "CubeFactory.h"
#include "Cube.h"
#include "Cube_C.h"
#include "ActorSet.h"

class Stage : public QGLWidget
{
  Q_OBJECT

public:
  Stage(QWidget* parent = 0);
  ActorSet* actorSet;

protected:
  void initializeGL();
  void resizeGL(int width, int height);
  void paintGL();
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);
  void dragEnterEvent(QDragEnterEvent *event);
  void dropEvent(QDropEvent *event);

private:
  void draw();

  //Display Variables for 3D
  GLfloat rotationX;
  GLfloat rotationY;
  GLfloat rotationZ;
  GLfloat fovy3D;
  GLfloat aspect3D;
  GLfloat zNear3D;
  GLfloat zFar3D;
  GLfloat disp3DX;
  GLfloat disp3DY;
  GLfloat disp3DZ;

  QPoint  lastPos;

  QColor faceColors[4];
};


#endif

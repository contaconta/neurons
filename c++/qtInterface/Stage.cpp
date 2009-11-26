

#include "Stage.h"

#include <QtGui>
#include <QtOpenGL>



Stage::Stage(QWidget* parent) : QGLWidget(parent)
{

  setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));
  // Camera parameters
  fovy3D = 30;
  aspect3D = 1;
  zNear3D = 1;
  zFar3D = 1000;

  // Position of the origin of coordinates
  rotationX = 0;
  rotationY = 0;
  rotationZ = 0;
  disp3DX = 0;
  disp3DY = 0;
  disp3DZ = -500;

  actors.push_back(new Axis());

  setFocus();

  faceColors[0] = Qt::red;
  faceColors[1] = Qt::green;
  faceColors[2] = Qt::blue;
  faceColors[3] = Qt::yellow;
}

void Stage::initializeGL()
{
  GLenum err = glewInit();
  if (GLEW_OK != err)
    {
      /* Problem: glewInit failed, something is seriously wrong. */
      fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
  else {
    fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
  }

  Cube_P* cube = CubeFactory::load("/media/neurons/n7/3d/4/N7_4.nfo");
  cube->load_texture_brick(0,0);
  actors.push_back(cube);

  qglClearColor(Qt::white);
  glShadeModel(GL_FLAT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
}

void Stage::resizeGL(int width, int height)
{
  glViewport(0,0,width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fovy3D, aspect3D, zNear3D, zFar3D);
  glScalef(1.0,1.0,1.0);
}

void Stage::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  draw();
}


void Stage::mousePressEvent(QMouseEvent *event)
{
  lastPos = event->pos();
  setFocus();
}

void Stage::mouseMoveEvent(QMouseEvent *event)
{
  GLfloat dx = GLfloat(event->x() - lastPos.x()) / width();
  GLfloat dy = GLfloat(event->y() - lastPos.y()) / height();

  if (event->buttons() & Qt::LeftButton) {
    rotationX += 200 * dy;
    rotationY += 200 * dx;
    updateGL();
  } else if (event->buttons() & Qt::RightButton) {
    disp3DZ += 150 * dx;
    updateGL();
  } else if (event->buttons() & Qt::MidButton) {
    disp3DX += 60 * dx;
    disp3DY -= 60 * dy;
    updateGL();
  }
  lastPos = event->pos();
}

void Stage::mouseDoubleClickEvent(QMouseEvent *event)
{
}

void Stage::draw()
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScalef(1, 1, 1);
  glTranslatef(disp3DX,disp3DY,disp3DZ);
  glRotatef(rotationX,1,0,0);
  glRotatef(rotationY,0,1,0);

  for(int i = 0; i < actors.size(); i++)
    actors[i]->draw();
}

void Stage::keyPressEvent(QKeyEvent *event)
{
  float angleStep   = 2;
  float dispStepXY  = 2;
  float dispStepZ   = 5;
  switch(event->key())
    {
    case Qt::Key_Left:
      switch(event->modifiers()){
      case Qt::NoModifier:
        rotationY -= angleStep;
        break;
      case Qt::ShiftModifier:
        disp3DX -= dispStepXY;
        break;
      }
      break;
    case Qt::Key_Right:
      switch(event->modifiers()){
      case Qt::NoModifier:
        rotationY += angleStep;
        break;
      case Qt::ShiftModifier:
        disp3DX += dispStepXY;
        break;
      }
      break;
    case Qt::Key_Up:
      switch(event->modifiers()){
      case Qt::NoModifier:
        rotationX -= angleStep;
        break;
      case Qt::ShiftModifier:
        disp3DY += dispStepXY;
        break;
      case Qt::ControlModifier:
        disp3DZ += dispStepZ;
        break;
      }
      break;
    case Qt::Key_Down:
      switch(event->modifiers()){
      case Qt::NoModifier:
        rotationX += angleStep;
        break;
      case Qt::ShiftModifier:
        disp3DY -= dispStepXY;
        break;
      case Qt::ControlModifier:
        disp3DZ -= dispStepZ;
        break;
      }
      break;
    case Qt::Key_Escape:
      printf("No exit in here has been pressed\n");
      break;
    default:
      QGLWidget::keyPressEvent(event);
    }
  updateGL();
}


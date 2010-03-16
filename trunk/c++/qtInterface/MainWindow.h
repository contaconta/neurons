#ifndef MAINWINDOWVIVA_H_
#define MAINWINDOWVIVA_H_

#include "ui_main.h"
#include "CubeDialog.h"
#include "Stage.h"
#include <QtGui/QFileDialog>

class MainWindow: public QMainWindow, Ui::MainWindow
{

  Q_OBJECT

public:
  MainWindow();
  void setActorSet(ActorSet* newSet);


private slots:
  void open();
  void setView3D();
  void setViewXY();
  void setViewXZ();
  void setViewYZ();
  void setViewCombo();

private:
  CubeDialog *cubeDialog;
  ActorSet* actorSet; //a pointer to all the objects to be drawn

  // void createActions();
  // void createMenus();
  void connectActionsToSlots();


};




#endif

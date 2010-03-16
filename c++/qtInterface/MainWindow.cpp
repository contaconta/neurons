#include "MainWindow.h"

MainWindow::MainWindow() : QMainWindow(), Ui::MainWindow(){
    setupUi(this);
    actorSet = new ActorSet();
    graphicsView->actorSet = actorSet;
    connectActionsToSlots();

    // cubeDialog = new CubeDialog();
    // cubeDialog->setObjectName(QString::fromUtf8("cubeDialog"));
    // cubeDialog->setGeometry(QRect(0, 20, 251, 381));
    // cubeDialog->setFrameShape(QFrame::StyledPanel);
    // cubeDialog->setFrameShadow(QFrame::Raised);
    // tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Cube", 0, QApplication::UnicodeUTF8));
    // cubeDialog->
  }

void MainWindow::connectActionsToSlots(){
  connect(actionAdd_Object, SIGNAL(triggered()), this, SLOT(open()));
  connect(action3D, SIGNAL(triggered()), this, SLOT(setView3D()));
  connect(actionXY, SIGNAL(triggered()), this, SLOT(setViewXY()));
  connect(actionXZ, SIGNAL(triggered()), this, SLOT(setViewXZ()));
  connect(actionYZ, SIGNAL(triggered()), this, SLOT(setViewYZ()));
  connect(actionCombo, SIGNAL(triggered()), this, SLOT(setViewCombo()));
}

void MainWindow::setActorSet(ActorSet* newSet){
  actorSet = newSet;
  graphicsView->actorSet = newSet;
}

void MainWindow::open()
{
  QString fileName = QFileDialog::getOpenFileName(this);
  if (!fileName.isEmpty())
    actorSet->addActorFromPath(qPrintable(fileName));


  printf("And now all the dialog stuff\n");
}
void MainWindow::setView3D()
{
  printf("setView3D\n");
}
void MainWindow::setViewXY()
{
  printf("setViewXY\n");
}
void MainWindow::setViewXZ()
{
  printf("setViewXZ\n");
}
void MainWindow::setViewYZ()
{
  printf("setViewXZ\n");
}
void MainWindow::setViewCombo()
{
  printf("setViewCombo\n");
}

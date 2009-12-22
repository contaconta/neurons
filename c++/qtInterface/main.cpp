#include <glew.h>
#include <QApplication>
#include "ui_main.h"
#include "CubeDialog.h"
#include "Stage.h"

class MainWindow: public QMainWindow, Ui::MainWindow
{
public:
  MainWindow() : QMainWindow(), Ui::MainWindow(){
    setupUi(this);
    // cubeDialog = new CubeDialog();

    // cubeDialog->setObjectName(QString::fromUtf8("cubeDialog"));
    // cubeDialog->setGeometry(QRect(0, 20, 251, 381));
    // cubeDialog->setFrameShape(QFrame::StyledPanel);
    // cubeDialog->setFrameShadow(QFrame::Raised);
    // tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Cube", 0, QApplication::UnicodeUTF8));

    // cubeDialog->show();
  }

  CubeDialog *cubeDialog;
};




 int main(int argc, char *argv[])
 {
     QApplication app(argc, argv);
     glutInit(&argc, argv);

     MainWindow mainWin;
     mainWin.show();
     return app.exec();
 }

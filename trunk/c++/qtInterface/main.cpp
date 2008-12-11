#include <glew.h>
#include <QApplication>
#include "ui_main.h"

class MainWindow: public QMainWindow, Ui::MainWindow
{
public:
  MainWindow() : QMainWindow(), Ui::MainWindow(){
    setupUi(this);
  }

};




 int main(int argc, char *argv[])
 {
     QApplication app(argc, argv);
     glutInit(&argc, argv);
  // int win = glutCreateWindow("GLEW Test");
  // glutDestroyWindow(win);

     MainWindow mainWin;
     mainWin.show();
     return app.exec();
 }

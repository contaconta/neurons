#include <glew.h>
#include <QApplication>
#include "ActorSet.h"
#include "MainWindow.h"



 int main(int argc, char *argv[])
 {
     QApplication app(argc, argv);
     glutInit(&argc, argv);

     MainWindow mainWin;
     mainWin.show();


     ActorSet* aset = new ActorSet();;
     for(int i = 1; i < argc; i++){
       aset->addActorFromPath(argv[i]);
     }
     mainWin.setActorSet(aset);


     return app.exec();
 }

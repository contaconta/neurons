/********************************************************************************
** Form generated from reading ui file 'main.ui'
**
** Created: Thu Jan 7 02:16:08 2010
**      by: Qt User Interface Compiler version 4.5.2
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_MAIN_H
#define UI_MAIN_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QWidget>
#include "Stage.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionAdd_Object;
    QAction *action3D;
    QAction *actionCombo;
    QAction *actionXY;
    QAction *actionXZ;
    QAction *actionYZ;
    QAction *actionScreen_shot;
    QAction *actionScreenShot;
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout;
    Stage *graphicsView;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuView;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 600);
        MainWindow->setFocusPolicy(Qt::StrongFocus);
        MainWindow->setAcceptDrops(false);
        QIcon icon;
        icon.addFile(QString::fromUtf8("../../../.designer/assets/icon.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        actionAdd_Object = new QAction(MainWindow);
        actionAdd_Object->setObjectName(QString::fromUtf8("actionAdd_Object"));
        action3D = new QAction(MainWindow);
        action3D->setObjectName(QString::fromUtf8("action3D"));
        actionCombo = new QAction(MainWindow);
        actionCombo->setObjectName(QString::fromUtf8("actionCombo"));
        actionXY = new QAction(MainWindow);
        actionXY->setObjectName(QString::fromUtf8("actionXY"));
        actionXZ = new QAction(MainWindow);
        actionXZ->setObjectName(QString::fromUtf8("actionXZ"));
        actionYZ = new QAction(MainWindow);
        actionYZ->setObjectName(QString::fromUtf8("actionYZ"));
        actionScreen_shot = new QAction(MainWindow);
        actionScreen_shot->setObjectName(QString::fromUtf8("actionScreen_shot"));
        actionScreenShot = new QAction(MainWindow);
        actionScreenShot->setObjectName(QString::fromUtf8("actionScreenShot"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        centralwidget->setFocusPolicy(Qt::StrongFocus);
        horizontalLayout = new QHBoxLayout(centralwidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        graphicsView = new Stage(centralwidget);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));
        sizePolicy.setHeightForWidth(graphicsView->sizePolicy().hasHeightForWidth());
        graphicsView->setSizePolicy(sizePolicy);
        graphicsView->setAutoFillBackground(true);

        horizontalLayout->addWidget(graphicsView);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 25));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuView = new QMenu(menubar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        MainWindow->setMenuBar(menubar);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuView->menuAction());
        menuFile->addAction(actionAdd_Object);
        menuView->addAction(action3D);
        menuView->addAction(actionCombo);
        menuView->addAction(actionXY);
        menuView->addAction(actionXZ);
        menuView->addAction(actionYZ);
        menuView->addSeparator();
        menuView->addSeparator();
        menuView->addAction(actionScreenShot);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "VIVA - Viewer", 0, QApplication::UnicodeUTF8));
        actionAdd_Object->setText(QApplication::translate("MainWindow", "Open", 0, QApplication::UnicodeUTF8));
        action3D->setText(QApplication::translate("MainWindow", "3D", 0, QApplication::UnicodeUTF8));
        actionCombo->setText(QApplication::translate("MainWindow", "Combo", 0, QApplication::UnicodeUTF8));
        actionXY->setText(QApplication::translate("MainWindow", "XY", 0, QApplication::UnicodeUTF8));
        actionXZ->setText(QApplication::translate("MainWindow", "XZ", 0, QApplication::UnicodeUTF8));
        actionYZ->setText(QApplication::translate("MainWindow", "YZ", 0, QApplication::UnicodeUTF8));
        actionScreen_shot->setText(QApplication::translate("MainWindow", "Screen-shot", 0, QApplication::UnicodeUTF8));
        actionScreenShot->setText(QApplication::translate("MainWindow", "ScreenShot", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", 0, QApplication::UnicodeUTF8));
        menuView->setTitle(QApplication::translate("MainWindow", "View", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAIN_H

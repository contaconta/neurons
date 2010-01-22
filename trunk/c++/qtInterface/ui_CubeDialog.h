/********************************************************************************
** Form generated from reading ui file 'CubeDialog.ui'
**
** Created: Thu Jan 7 02:16:08 2010
**      by: Qt User Interface Compiler version 4.5.2
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_CUBEDIALOG_H
#define UI_CUBEDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCommandLinkButton>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CubeDialog
{
public:
    QGroupBox *groupBox;
    QRadioButton *radioButtonMaximum;
    QPushButton *pushButton;
    QCommandLinkButton *commandLinkButton;
    QLineEdit *lineEdit;

    void setupUi(QWidget *CubeDialog)
    {
        if (CubeDialog->objectName().isEmpty())
            CubeDialog->setObjectName(QString::fromUtf8("CubeDialog"));
        CubeDialog->resize(222, 733);
        groupBox = new QGroupBox(CubeDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(0, 10, 221, 51));
        radioButtonMaximum = new QRadioButton(groupBox);
        radioButtonMaximum->setObjectName(QString::fromUtf8("radioButtonMaximum"));
        radioButtonMaximum->setGeometry(QRect(10, 20, 92, 24));
        radioButtonMaximum->setChecked(true);
        pushButton = new QPushButton(CubeDialog);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setGeometry(QRect(10, 70, 92, 28));
        commandLinkButton = new QCommandLinkButton(CubeDialog);
        commandLinkButton->setObjectName(QString::fromUtf8("commandLinkButton"));
        commandLinkButton->setGeometry(QRect(10, 130, 179, 41));
        lineEdit = new QLineEdit(CubeDialog);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        lineEdit->setGeometry(QRect(20, 190, 113, 26));
        QWidget::setTabOrder(radioButtonMaximum, pushButton);
        QWidget::setTabOrder(pushButton, commandLinkButton);

        retranslateUi(CubeDialog);

        QMetaObject::connectSlotsByName(CubeDialog);
    } // setupUi

    void retranslateUi(QWidget *CubeDialog)
    {
        CubeDialog->setWindowTitle(QApplication::translate("CubeDialog", "Cube", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("CubeDialog", "Projection", 0, QApplication::UnicodeUTF8));
        radioButtonMaximum->setText(QApplication::translate("CubeDialog", "Max IP", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("CubeDialog", "PushButton", 0, QApplication::UnicodeUTF8));
        commandLinkButton->setText(QApplication::translate("CubeDialog", "CommandLinkButton", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(CubeDialog);
    } // retranslateUi

};

namespace Ui {
    class CubeDialog: public Ui_CubeDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CUBEDIALOG_H

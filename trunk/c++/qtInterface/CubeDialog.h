#ifndef CUBEDIALOG_H_
#define CUBEDIALOG_H_

#include <QFrame>
#include "ui_CubeDialog.h"

class CubeDialog : public QFrame, Ui::CubeDialog
{
  Q_OBJECT

public:
  CubeDialog(QWidget* parent = 0);
};


#endif

/****************************************************************************
** Meta object code from reading C++ file 'CubeDialog.h'
**
** Created: Thu Jan 7 02:16:13 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "CubeDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'CubeDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_CubeDialog[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

       0        // eod
};

static const char qt_meta_stringdata_CubeDialog[] = {
    "CubeDialog\0"
};

const QMetaObject CubeDialog::staticMetaObject = {
    { &QFrame::staticMetaObject, qt_meta_stringdata_CubeDialog,
      qt_meta_data_CubeDialog, 0 }
};

const QMetaObject *CubeDialog::metaObject() const
{
    return &staticMetaObject;
}

void *CubeDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CubeDialog))
        return static_cast<void*>(const_cast< CubeDialog*>(this));
    if (!strcmp(_clname, "Ui::CubeDialog"))
        return static_cast< Ui::CubeDialog*>(const_cast< CubeDialog*>(this));
    return QFrame::qt_metacast(_clname);
}

int CubeDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QFrame::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE

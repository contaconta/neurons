/***************************************************************************
                          LSMove.h  -  description
                             -------------------
    begin                : Tue Sept 25 2001
    copyright            : (C) 2001 by Christian Blum
    email                : cblum@ulb.ac.be
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef LSMOVE_H
#define LSMOVE_H

#include "config.h"

#include "Leaf.h"

/**
  *@author Christian Blum
  */

class LSMove {
public: 
  Leaf* in;
  Leaf* out;
  double weight_diff;

  LSMove();
  LSMove(Leaf* in_comp, Leaf* out_comp);
  ~LSMove();

  LSMove* copy();
};

#endif

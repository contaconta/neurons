/***************************************************************************
                          LSMove.cpp  -  description
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

#include "LSMove.h"

LSMove::LSMove(){
}

LSMove::LSMove(Leaf* in_comp, Leaf* out_comp) {

  in = in_comp;
  out = out_comp;
  weight_diff = (in->lEdge)->weight() - (out->lEdge)->weight();
}

LSMove::~LSMove(){
}

LSMove* LSMove::copy() {

  LSMove* copy = new LSMove(in->copy(),out->copy());
  return copy;
}

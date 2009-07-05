/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by Aurelien Lucchi and Kevin Smith                  //
// Contact aurelien.lucchi (at) gmail.com or kevin.smith (at) epfl.ch  // 
// for comments & bug reports                                          //
/////////////////////////////////////////////////////////////////////////

#include <vector>

using namespace std;

// Create a list of all the possible combination of weak learners for the specified learner type
// @param max_width : maximum width of the weak learners
// @param max_height : maximum height of the weak learners
// @param weak_learners is an double array containing the list of weak learners
//        The memory for this array has to be allocated inside the function as
//        we don't know the size needed before running the function.
// @param weak_learner_type_indices has to be an array of size nb_learner_type
//        This functions will populate this array with the first indices of
//        each learner type in the array weak_learner
int enumerate_learners(char **learner_type, int nb_learner_type,
                       int max_width, int max_height,
                       char**& weak_learners, int* weak_learner_type_indices);

#include "Axis.h"

void Axis::draw(){
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0,0,0);
    glVertex3f(0,0, 100);
    glEnd();
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0,0,0);
    glVertex3f(100,0,0);
    glEnd();
    glBegin(GL_LINES);
    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0,0,0);
    glVertex3f(0,100000, 0);
    glEnd();

    glColor3f(1.0, 1.0, 0.0);
    glutSolidSphere(1.0, 20, 20);
    glPushMatrix();

      //Draw The Z axis
    glColor3f(0.0, 0.0, 1.0);
    glRotatef(180, 1.0, 0, 0);
    glutSolidCone(2.0, 10, 20, 20);
    // glBegin(GL_LINES);
    // glVertex3f(0,0,0);
    // glVertex3f(0,0, 100000);
    // glEnd();
    
      //Draw the x axis
    glColor3f(1.0, 0.0, 0.0);
    glRotatef(90, 0.0, 1.0, 0.0);
    glutSolidCone(2.0, 10, 20, 20);
    // glBegin(GL_LINES);
    // glVertex3f(0,0,0);
    // glVertex3f(0,0, 100000);
    // glEnd();
    
      //Draw the y axis
    glColor3f(0.0, 1.0, 0.0);
    glRotatef(90, 1.0, 0.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    // glBegin(GL_LINES);
    // glVertex3f(0,0,0);
    // glVertex3f(0,0, 100000);
    // glEnd();
    glPopMatrix();
    glColor3f(1.0, 1.0, 1.0);
  }

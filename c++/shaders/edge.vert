void main(void)
{
   //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   gl_Position = ftransform();
   
   gl_FrontColor = gl_Color;
   gl_BackColor = gl_Color;
	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
} 

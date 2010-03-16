uniform sampler3D texUnit;

void main(void)
{
  // ---------------
  // | tl | t | ur |
  // | l  | c | r  |
  // | bl | b | br |
  // ---------------
  const float offset = 1.0 / 512.0;
  vec3 texCoord = gl_TexCoord[0].xyz;
  vec4 c  = texture3D(texUnit, texCoord);
  vec4 bl = texture3D(texUnit, texCoord + vec3(-offset, -offset,     0.0));
  vec4 l  = texture3D(texUnit, texCoord + vec3(-offset,     0.0,     0.0));
  vec4 tl = texture3D(texUnit, texCoord + vec3(-offset,  offset,     0.0));
  vec4 t  = texture3D(texUnit, texCoord + vec3(    0.0,  offset,     0.0));
  vec4 ur = texture3D(texUnit, texCoord + vec3( offset,  offset,     0.0));
  vec4 r  = texture3D(texUnit, texCoord + vec3( offset,     0.0,     0.0));
  vec4 br = texture3D(texUnit, texCoord + vec3( offset,  offset,     0.0));
  vec4 b  = texture3D(texUnit, texCoord + vec3(    0.0, -offset,     0.0));

  // Discard specific colors
  //if (c.rgb == vec3(1.0,0.0,0.0))
  if(c.r > 0.5f)
    discard;
  else
    gl_FragColor = c;
}

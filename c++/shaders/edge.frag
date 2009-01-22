uniform sampler3D texUnit;
uniform int filterId;

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

	if(filterId == 1)
	{
		// Invert color
		gl_FragColor = 1 - c;
	}
	else if(filterId == 2)
	{
		// Laplace operator
		gl_FragColor = 8.0 * (c + -0.125 * (bl + l + tl + t + ur + r + br + b));
	}
	else if(filterId == 3)
	{
		// Sobel operator Gx
		vec4 gx = bl + 2.0 * l + tl - ur - 2.0 * r - br;
		// Sobel operator Gy
		vec4 gy = bl + 2.0 * b + br - ur - 2.0 * t - tl;
		// Sobel operator G = (Gx² + Gy²)^0.5
		gl_FragColor = sqrt(gx * gx + gy * gy);
	}
	else if(filterId == 4)
	{
		// Kirsch operator Gx
		vec4 gx = 5.0 * tl - 3.0 * l - 3.0 * bl - 3.0 * b - 3.0 * br - 3.0 * r + 5.0 * ur + 5.0 * t;
		// Kirsch operator Gy
		vec4 gy = -3.0 * tl - 3.0 * l - 3.0 * bl - 3.0 * b + 5.0 * br + 5.0 * r + 5.0 * ur - 3.0 * t;
		// Kirsch operator G = (Gx² + Gy²)^0.5
		gl_FragColor = sqrt(gx * gx + gy * gy);
	}
	else if(filterId == 5)
	{
		// Compass operator
		vec4 max = 0;
		vec4 Gi;
		float temp;
		// kernel coefficients
		// Set of 8 kernels is produced by taking one of the kernels
		// and rotating its coefficients circularly.
		// Each of the resulting kernels is sensitive to an edge orientation
		// ranging from 0° to 315° in steps of 45°, where 0° corresponds to a vertical edge. 
		// Coefficient in the center will never change, always -2
		float coeffs[8] = {-1,-1,-1,1,1,1,1,1};
		for(int i=0;i<8;i++)
		{
			Gi = - 2.0 * c + coeffs[0] * tl + coeffs[1] * l;
			Gi	+= coeffs[2] * bl + coeffs[3] * b;
			Gi	+= coeffs[4] * br + coeffs[5] * r;
			Gi	+= coeffs[6] * ur + coeffs[7] * t;

			if((max.x+max.y+max.z)<(Gi.x+Gi.y+Gi.z))
				max = Gi;

			// Right shifting
			if(i!=7)
			{
				temp = coeffs[7];
				for(int j=7;j>0;j--)
					coeffs[j] = coeffs[j-1];
				coeffs[0] = temp;
			}
		}
		gl_FragColor = max;
	}
	else if(filterId == 6)
	{
		// Gaussian
		gl_FragColor = 0.0625 * (4.0 * c + bl + 2.0 * l + tl + 2.0 * t + ur + 2.0 * r + br + 2.0 * b);
	}
}

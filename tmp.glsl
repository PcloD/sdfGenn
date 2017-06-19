int binary_search(int code, float sdfs[8])
{
    if (code==(uint)134217731)
    {
        sdfs[0] = 0.000000;
        sdfs[1] = 0.414213;
        sdfs[2] = 0.417062;
        sdfs[3] = 0.734936;
        sdfs[4] = -0.995472;
        sdfs[5] = 0.000000;
        sdfs[6] = 0.000000;
        sdfs[7] = 0.416192;
        return 3;
    }
    else if (code<(uint)134217731)
    { 
          if (code==(uint)134217729)
          {
              sdfs[0] = 0.417062;
              sdfs[1] = 0.732864;
              sdfs[2] = 0.000000;
              sdfs[3] = 0.414213;
              sdfs[4] = 0.000000;
              sdfs[5] = 0.416382;
              sdfs[6] = -0.995472;
              sdfs[7] = 0.000000;
              return 1;
          }
          else if (code<(uint)134217729)
          { 
                  if (code==(uint)134217728)
                  {
                      sdfs[0] = 0.734936;
                      sdfs[1] = 0.417062;
                      sdfs[2] = 0.414213;
                      sdfs[3] = 0.000000;
                      sdfs[4] = 0.416192;
                      sdfs[5] = 0.000000;
                      sdfs[6] = 0.000000;
                      sdfs[7] = -0.995472;
                      return 0;
                  }
                  else return -1; 
          }
          else 
          { 
                  if (code==(uint)134217730)
                  {
                      sdfs[0] = 0.414213;
                      sdfs[1] = 0.000000;
                      sdfs[2] = 0.732864;
                      sdfs[3] = 0.417062;
                      sdfs[4] = 0.000000;
                      sdfs[5] = -0.995472;
                      sdfs[6] = 0.416382;
                      sdfs[7] = 0.000000;
                      return 2;
                  }
                  else return -1; 
          }
    }
    else 
    { 
          if (code==(uint)134217733)
          {
              sdfs[0] = 0.000000;
              sdfs[1] = 0.416382;
              sdfs[2] = -0.995472;
              sdfs[3] = 0.000000;
              sdfs[4] = 0.417062;
              sdfs[5] = 0.732864;
              sdfs[6] = 0.000000;
              sdfs[7] = 0.414213;
              return 5;
          }
          else if (code<(uint)134217733)
          { 
                  if (code==(uint)134217732)
                  {
                      sdfs[0] = 0.416192;
                      sdfs[1] = 0.000000;
                      sdfs[2] = 0.000000;
                      sdfs[3] = -0.995472;
                      sdfs[4] = 0.734936;
                      sdfs[5] = 0.417062;
                      sdfs[6] = 0.414213;
                      sdfs[7] = 0.000000;
                      return 4;
                  }
                  else return -1; 
          }
          else 
          { 
                  if (code==(uint)134217734)
                  {
                      sdfs[0] = 0.000000;
                      sdfs[1] = -0.995472;
                      sdfs[2] = 0.416382;
                      sdfs[3] = 0.000000;
                      sdfs[4] = 0.414213;
                      sdfs[5] = 0.000000;
                      sdfs[6] = 0.732864;
                      sdfs[7] = 0.417062;
                      return 6;
                  }
                  else if (code<(uint)134217734)
                  { 
                            if (code==(uint)134217734)
                            {
                                sdfs[0] = 0.000000;
                                sdfs[1] = -0.995472;
                                sdfs[2] = 0.416382;
                                sdfs[3] = 0.000000;
                                sdfs[4] = 0.414213;
                                sdfs[5] = 0.000000;
                                sdfs[6] = 0.732864;
                                sdfs[7] = 0.417062;
                                return 6;
                            }
                            else return -1; 
                  }
                  else 
                  { 
                            if (code==(uint)134217735)
                            {
                                sdfs[0] = -0.995472;
                                sdfs[1] = 0.000000;
                                sdfs[2] = 0.000000;
                                sdfs[3] = 0.416192;
                                sdfs[4] = 0.000000;
                                sdfs[5] = 0.414213;
                                sdfs[6] = 0.417062;
                                sdfs[7] = 0.734936;
                                return 7;
                            }
                            else return -1; 
                  }
          }
    }
}

#define MAX_DEPTH 4


vec3 nodeOrigin(uint code)
{
	
}

float SDF(in vec3 X)
{
	vec3 O = vec3(-1.000000, -1.000000, -1.000000);
	float edge = 2.0;
	vec3 lsP = (X - O)/edge;

	return edge;

	/*
	for (int depth=0; depth<MAX_DEPTH; ++depth)
	{
		uint code = genNodeCode(lsP, depth);
		float sdfs[8];
		if ( binary_search(code, sdfs) >= 0 )
		{
			vec3 box_origin = nodeOrigin(code);
			vec3 f2 = (lsP - box_origin) * invNodeSize(nodeDepth(code));
			vec3 f1 = vec3(1.0) - f2;

			float mmm = sdfs[0]; // x=f1, y=f1, z=f1
			float pmm = sdfs[1]; // x=f2, y=f1, z=f1
			float mpm = sdfs[2]; // x=f1, y=f2, z=f1
			float ppm = sdfs[3]; // x=f2, y=f2, z=f1
			float mmp = sdfs[4]; // x=f1, y=f1, z=f2
			float pmp = sdfs[5]; // x=f2, y=f1, z=f2
			float mpp = sdfs[6]; // x=f1, y=f2, z=f2
			float ppp = sdfs[7]; // x=f2, y=f2, z=f2

			return ( f1.x * (f1.y*(f1.z*mmm + f2.z*mmp)  +
			                 f2.y*(f1.z*mpm + f2.z*mpp)) +
			         f2.x * (f1.y*(f1.z*pmm + f2.z*pmp)  +
			                 f2.y*(f1.z*ppm + f2.z*ppp)) );
		}
	}

	return 1.0e6;
	*/
}


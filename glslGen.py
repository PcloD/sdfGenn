
import string

########################## Python code #################################

octree_origin = [-1.000000, -1.000000, -1.000000]
octree_edge = 2.000000
morton_codes = [134217728, 134217729, 134217730, 134217731, 134217732, 134217733, 134217734, 134217735]
sdfs = [0.734936, 0.417062, 0.414213, 0.000000, 0.416192, 0.000000, 0.000000, -0.995472, 0.417062, 0.732864, 0.000000, 0.414213, 0.000000, 0.416382, -0.995472, 0.000000, 0.414213, 0.000000, 0.732864, 0.417062, 0.000000, -0.995472, 0.416382, 0.000000, 0.000000, 0.414213, 0.417062, 0.734936, -0.995472, 0.000000, 0.000000, 0.416192, 0.416192, 0.000000, 0.000000, -0.995472, 0.734936, 0.417062, 0.414213, 0.000000, 0.000000, 0.416382, -0.995472, 0.000000, 0.417062, 0.732864, 0.000000, 0.414213, 0.000000, -0.995472, 0.416382, 0.000000, 0.414213, 0.000000, 0.732864, 0.417062, -0.995472, 0.000000, 0.000000, 0.416192, 0.000000, 0.414213, 0.417062, 0.734936]

########################## Python code  end #################################


def indent(s, num_spaces):
	s.lstrip('\n').rstrip(' \t')
	indent = ' '*num_spaces
	s = indent + s.replace('\n', '\n'+indent)
	return s

def sdf_code(leaf_index):
	sdf_start = 8*leaf_index
	glsl = ''
	for n in range(0, 8):
		s = sdfs[sdf_start+n]
		glsl += '    sdfs[%d] = %f;\n' % (n, s)
	return glsl


def descend(depth, left, right):

	if left>=right: 
		return indent('''
if (code==(uint)%d)
{
%s  
    return %d;
}
else return -1;''' % (morton_codes[left], sdf_code(left), left), 4+depth*2)
	
	middle = int(left+right)/2
	middleVal = morton_codes[middle]
	glsl = string.Template('''
if (code==(uint)$middleVal)
{
${SDF_MIDDLE}
    return $middle;
}
else if (code<(uint)$middleVal)
{ 
    $${NEXT_LEFT} 
}
else 
{ 
    $${NEXT_RIGHT} 
}
''').safe_substitute({'left': left,
                      'right': right,
		              'middle': middle,
		              'middle_m': middle-1,
		              'middle_p': middle+1,
		              'middleVal': middleVal,
		              'SDF_MIDDLE': sdf_code(middle)})

	glsl_left  = descend(depth+1, left, middle-1)
	glsl_right = descend(depth+1, middle+1, right)
	glsl = string.Template(glsl).safe_substitute({'NEXT_LEFT': glsl_left, 'NEXT_RIGHT': glsl_right})

	return indent(glsl, 4+depth*2)

left = 0
right = len(morton_codes)-1


glsl =  string.Template('''
uint binary_search(int code, float sdfs[8])
{
	${func}
}
''').safe_substitute({'func': descend(0, left, right)})
glsl = "".join([s for s in glsl.splitlines(True) if s.strip()])

glsl += string.Template('''
#define MAX_DEPTH 4


vec3 nodeOrigin(uint code)
{
	
}

float SDF(in vec3 X)
{
	vec3 O = vec3(${ORIGIN});
	float edge = ${EDGE};
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
''').safe_substitute( {'ORIGIN': '%f, %f, %f'%(octree_origin[0], octree_origin[1], octree_origin[2]),
                       'EDGE': octree_edge
	                      } )

print glsl







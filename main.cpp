
#include <iostream>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <algorithm>
#include <vector>
#include <stack>
#include <bitset>
#include <unordered_map>
#include <queue>
#include <limits.h>

using namespace std;

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#include <mmsystem.h>
#include <windows.h>
#ifdef __cplusplus
}
#endif
#pragma comment(lib, "winmm.lib")
#else
#if defined(__unix__) || defined(__APPLE__)
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#else
#include <ctime>
#endif
#endif

class Vec4
{
public:

    Vec4();										     // Initializes Vec4 to (0, 0, 0, 1)

    Vec4 (double x);                                    // Initializes Vec4 to (x, x, x, 1)
    Vec4( double x, double y, double z, double w=1.f ); // Initializes Vec4 to (x, y, z, w)
    Vec4 (const double *v);                             // Initializes Vec4 to (v[0], v[1], v[2], 1)
    Vec4 (const float *v);                              // Initializes Vec4 to (v[0], v[1], v[2], 1)
    Vec4( const Vec4& other );						    // copy constructor

    // assignment
    Vec4& operator=(const Vec4 &rhs)
    {
        m_v[0] = rhs.m_v[0];
        m_v[1] = rhs.m_v[1];
        m_v[2] = rhs.m_v[2];
        m_v[3] = rhs.m_v[3];
        return *this;
    }

    inline double x() const { return m_v[0]; }
    inline double y() const { return m_v[1]; }
    inline double z() const { return m_v[2]; }
    inline double w() const { return m_v[3]; }

    // access to elements
    double &operator[](const size_t);
    const double &operator[](const size_t) const;

    // magnitude
    double length3(void) const;
    double length3Sqr(void) const;

    // normalize
    void normalizeIfNotZero();

    // zero
    void setZero3();
    void setZero4();

    // index and value of maximum element
    int maxIndex(double& maxValue) const;

    // Return the component-wise product of A and B
    static Vec4 directProduct(const Vec4& A, const Vec4& B);

    // addition and subtraction
    friend Vec4& operator+=( Vec4& lhs, const Vec4& rhs );
    friend Vec4 operator+( const Vec4& lhs, const Vec4& rhs );

    friend Vec4& operator-=( Vec4& lhs, const Vec4& rhs );
    friend Vec4 operator-( const Vec4& lhs, const Vec4& rhs );

    // multiplication and division by scalars
    friend Vec4& operator*=( Vec4& lhs, const double rhs );
    friend Vec4 operator*( const Vec4& lhs, const double rhs );
    friend Vec4 operator*( const double lhs, const Vec4& rhs );

    friend Vec4& operator/=( Vec4& lhs, const double rhs );
    friend Vec4 operator/( const Vec4& lhs, const double rhs );

    // dot product
    friend double operator*( const Vec4& lhs, const Vec4& rhs );
    friend double operator^( const Vec4& lhs, const Vec4& rhs );

    // cross product
    friend Vec4& operator%=( Vec4& lhs, const Vec4& rhs );
    friend Vec4 operator%( const Vec4& lhs, const Vec4& rhs );

    friend bool operator==( const Vec4& lhs, const Vec4& rhs );
    friend bool operator!=( const Vec4& lhs, const Vec4& rhs );


private:

    double m_v[4];

};


//element access
inline double& Vec4::operator[](const size_t index)
{
    return m_v[index];
}

inline const double& Vec4::operator[](const size_t index) const
{
    return m_v[index];
}


inline bool operator==( const Vec4& lhs, const Vec4& rhs )
{
    bool equal = true;
    for (int a=0; a<4; ++a)
    {
        equal &= ( lhs.m_v[a] == rhs.m_v[a] );
    }
    return equal;
}

inline bool operator!=( const Vec4& lhs, const Vec4& rhs )
{
    return !(lhs==rhs);
}

inline Vec4& operator+=( Vec4& lhs, const Vec4& rhs )
{
    for (int a=0; a<3; ++a)
    {
        lhs.m_v[a] += rhs.m_v[a];
    }
    return lhs;
}

inline Vec4& operator-=( Vec4& lhs, const Vec4& rhs )
{
    for (int a=0; a<3; ++a)
    {
        lhs.m_v[a] -= rhs.m_v[a];
    }
    return lhs;
}

inline Vec4 operator+( const Vec4& lhs, const Vec4& rhs )
{
    Vec4 ret(lhs);
    ret += rhs;
    return ret;
}

inline Vec4 operator-( const Vec4& lhs, const Vec4& rhs )
{
    Vec4 ret(lhs);
    ret -= rhs;
    return ret;
}

inline double operator*( const Vec4& lhs, const Vec4& rhs )
{
    double dot = 0.0;
    for (int a=0; a<3; ++a)
    {
        dot += lhs.m_v[a] * rhs.m_v[a];
    }
    return dot;
}

inline double operator^( const Vec4& lhs, const Vec4& rhs )
{
    double dot = 0.0;
    for (int a=0; a<3; ++a)
    {
        dot += lhs.m_v[a] * rhs.m_v[a];
    }
    return dot;
}

inline Vec4& operator%=( Vec4& lhs, const Vec4& rhs )
{
    Vec4 tmp = lhs;

    tmp.m_v[0] = lhs.m_v[1] * rhs.m_v[2] - lhs.m_v[2] * rhs.m_v[1];
    tmp.m_v[1] = lhs.m_v[2] * rhs.m_v[0] - lhs.m_v[0] * rhs.m_v[2];
    tmp.m_v[2] = lhs.m_v[0] * rhs.m_v[1] - lhs.m_v[1] * rhs.m_v[0];

    lhs = tmp;
    return lhs;
}

inline Vec4 operator%( const Vec4& lhs, const Vec4& rhs )
{
    Vec4 ret(lhs);
    ret %= rhs;
    return ret;
}

inline Vec4& operator*=( Vec4& lhs, const double rhs )
{
    for (int a=0; a<3; ++a)
    {
        lhs.m_v[a] *= rhs;
    }
    return lhs;
}

inline Vec4 operator*( const Vec4& lhs, const double rhs )
{
    Vec4 ret(lhs);
    ret *= rhs;
    return ret;
}
inline Vec4 operator*( const double lhs, const Vec4& rhs )
{
    Vec4 ret(rhs);
    ret *= lhs;
    return ret;
}

inline Vec4& operator/=( Vec4& lhs, const double rhs )
{
    if (rhs == 0.0) return lhs;
    for (int a=0; a<3; ++a)
    {
        lhs.m_v[a] /= rhs;
    }
    return lhs;
}

inline Vec4 operator/( const Vec4& lhs, const double rhs )
{
    Vec4 ret(lhs);
    if (rhs == 0.0) return lhs;
    ret /= rhs;
    return ret;
}

Vec4::Vec4()
{
    m_v[0] = 0.0;
    m_v[1] = 0.0;
    m_v[2] = 0.0;
    m_v[3] = 1.0;
}

Vec4::Vec4( double x )
{
    m_v[0] = x;
    m_v[1] = x;
    m_v[2] = x;
    m_v[3] = 1.0;
}

Vec4::Vec4( double x, double y, double z, double w )
{
    m_v[0] = x;
    m_v[1] = y;
    m_v[2] = z;
    m_v[3] = w;
}


Vec4::Vec4 (const double *v)
{
    m_v[0] = v[0];
    m_v[1] = v[1];
    m_v[2] = v[2];
    m_v[3] = 1.0;
}

Vec4::Vec4 (const float *v)
{
    m_v[0] = v[0];
    m_v[1] = v[1];
    m_v[2] = v[2];
    m_v[3] = 1.f;
}


// copy constructor
Vec4::Vec4( const Vec4& other )
{
    m_v[0] = other.m_v[0];
    m_v[1] = other.m_v[1];
    m_v[2] = other.m_v[2];
    m_v[3] = other.m_v[3];
}

//magnitude
double Vec4::length3(void) const
{
    return sqrt( (*this) * (*this) );
}

double Vec4::length3Sqr(void) const
{
    return (*this) * (*this);
}

//normalize
void Vec4::normalizeIfNotZero()
{
    double magsqr = (*this) * (*this);
    if (magsqr > 0.0)
    {
        if (magsqr == 1.0) return; // already normalized
        double mag = sqrt(magsqr);
        for (int a=0; a<3; ++a)
        {
            m_v[a] /= mag;
        }
    }
}

//zero x,y,z
void Vec4::setZero3()
{
    for (int a=0; a<3; ++a)
    {
        m_v[a] = 0.0;
    }
}

//zero x,y,z,w
void Vec4::setZero4()
{
    for (int a=0; a<4; ++a)
    {
        m_v[a] = 0.0;
    }
}


int Vec4::maxIndex(double& maxValue) const
{
    if (m_v[0] >= m_v[1])
    {
        if (m_v[0] >= m_v[2])
        {
            maxValue = m_v[0];
            return 0;
        }
        else
        {
            maxValue = m_v[2];
            return 2;
        }
    }
    else
    {
        if (m_v[1] >= m_v[2])
        {
            maxValue = m_v[1];
            return 1;
        }
        else
        {
            maxValue = m_v[2];
            return 2;
        }
    }
}

/* static */ Vec4 Vec4::directProduct(const Vec4& A, const Vec4& B)
{
    Vec4 v;
    for (int a=0; a<3; ++a)
    {
        v[a] = A.m_v[a] * B.m_v[a];
    }
    return v;
}


class Aabb
{

public:

    Aabb()
    {
        reset();
    }

    // Put the AABB in a state ready to include points.
    void reset();

    // Update this AABB to include the given point.
    void includePoint(const Vec4& point);
    void extendBy(const Vec4& point) { includePoint(point); }

    // Compute the AABB of the given vertices, overwriting this AABB.
    void compute( const vector<Vec4>& worldVerts );

    // Compute the union of this AABB with another one, overwriting this AABB.
    void add( const Aabb& aabbOther );
    void extendBy( const Aabb& aabbOther ) { add(aabbOther); }

    // Expand this AABB uniformly by a specified factor
    void expand( double factor );

    // Get center
    Vec4 center() const;

    bool intersects(const Aabb& aabbOther) const;

    // Get the total surface area of this AABB
    double surfaceArea( ) const;

    double volume( ) const;

    // Get the length of the largest edge
    double size() const;

    // Whether this AABB overlaps the other given one
    bool overlaps(const Aabb& other);

    // Debug draw
    void draw();

    int majorAxis() const;

    Vec4 m_maxima;
    Vec4 m_minima;

};

void Aabb::reset()
{
    const double HUGEVAL = 1.e20;
    m_maxima = Vec4(-HUGEVAL, -HUGEVAL, -HUGEVAL);
    m_minima = Vec4(HUGEVAL, HUGEVAL, HUGEVAL);
}

void Aabb::includePoint(const Vec4& point)
{
    for (int a=0; a<3; ++a)
    {
        if ( point[a] > m_maxima[a] ) m_maxima[a] = point[a];
        if ( point[a] < m_minima[a] ) m_minima[a] = point[a];
    }
}

void Aabb::compute( const vector<Vec4>& worldVerts )
{
    int nVerts = worldVerts.size();
    if (nVerts<1) return;

    m_maxima = worldVerts[0];
    m_minima = worldVerts[0];
    for (int n=1; n<nVerts; ++n)
    {
        const Vec4& verts = worldVerts[n];
        for (int a=0; a<3; ++a)
        {
            if ( verts[a] > m_maxima[a] ) m_maxima[a] = verts[a];
            if ( verts[a] < m_minima[a] ) m_minima[a] = verts[a];
        }
    }
}

int Aabb::majorAxis() const
{
    Vec4 diff = m_maxima - m_minima;
    double maxVal;
    return diff.maxIndex(maxVal);
}

void Aabb::add( const Aabb& aabbOther )
{
    const Vec4& maximaOther = aabbOther.m_maxima;
    const Vec4& minimaOther = aabbOther.m_minima;
    for (int a=0; a<3; ++a)
    {
        if ( maximaOther[a] > m_maxima[a] ) m_maxima[a] = maximaOther[a];
        if ( minimaOther[a] < m_minima[a] ) m_minima[a] = minimaOther[a];
    }
}

bool Aabb::intersects(const Aabb& aabbOther) const
{
    const Vec4& maximaOther = aabbOther.m_maxima;
    const Vec4& minimaOther = aabbOther.m_minima;
    for (int a=0; a<3; ++a)
    {
        if ( minimaOther[a] > m_maxima[a] ) return false;
        if ( maximaOther[a] < m_minima[a] ) return false;
    }
    return true;
}

void Aabb::expand( double factor )
{
    Vec4 diff = (m_maxima - m_minima) * (0.5 * factor);
    Vec4 center = (m_maxima + m_minima) * 0.5;

    m_maxima = center + diff;
    m_minima = center - diff;
}

Vec4 Aabb::center() const
{
    Vec4 center = (m_maxima + m_minima) * 0.5;
    return center;
}

double Aabb::surfaceArea() const
{
    Vec4 extents = m_maxima - m_minima;

    double A = 0.0;
    for (int axis=0; axis<3; ++axis)
    {
        int a = (axis+1)%3;
        int b = (axis+2)%3;
        A += 2.0 * extents[a] * extents[b];
    }

    return A;
}

double Aabb::volume() const
{
    Vec4 extents = m_maxima - m_minima;
    return extents[0] * extents[1] * extents[2];
}

// length of the largest edge
double Aabb::size() const
{
    Vec4 diff = m_maxima - m_minima;
    double maxVal;
    diff.maxIndex(maxVal);
    return maxVal;
}

bool Aabb::overlaps(const Aabb& other)
{
    for (int a=0; a<3; ++a)
    {
        if ( m_minima[a] > other.m_maxima[a] ) return false;
        if ( m_maxima[a] < other.m_minima[a] ) return false;
    }
    return true;
}

struct Geometry
{
    Geometry() : m_clockwiseWinding(true)
    {

    }

    vector<Vec4> m_vertices;
    vector<Vec4> m_vertexNormals;
    vector<Vec4> m_UVs;

    // 1 per-vertex, computed by summing normals of adjacent faces and re-normalizing
    vector<Vec4> m_smoothedNormals;

    void computeSmoothedNormals();

    struct Triangle
    {
        Triangle()
        {
            for (int a=0; a<3; ++a)
            {
                m_vertex[a] = -1;
                m_vertexNormal[a] = -1;
            }
        }

        int m_vertex[3];
        int m_vertexNormal[3];
        int m_uvs[3];
    };
    vector<Triangle> m_triangles;

    Aabb getTriangleAabb(int triIndex) const;

    // Needed to define outward normals, if no per-vertex normals
    bool m_clockwiseWinding;

    Aabb computeAabb() const;

};

Aabb Geometry::computeAabb() const
{
    Aabb aabb;
    for (int ti=0; ti<m_triangles.size(); ++ti)
    {
        aabb.add(getTriangleAabb(ti));
    }
    return aabb;
}

void Geometry::computeSmoothedNormals()
{
    Vec4 zeroVec; zeroVec.setZero4();
    m_smoothedNormals.resize(m_vertices.size(), zeroVec);

    for (int ti=0; ti<m_triangles.size(); ++ti)
    {
        Geometry::Triangle& triangle = m_triangles[ti];

        const Vec4& A = m_vertices[triangle.m_vertex[0]];
        const Vec4& B = m_vertices[triangle.m_vertex[1]];
        const Vec4& C = m_vertices[triangle.m_vertex[2]];

        Vec4 N = (B-A) % (C-A); // Assumed winding may be wrong, will fix shortly if so
        N.normalizeIfNotZero();

        // Weight normals by angle subtended at triangle vertex.
        Vec4 BA = B-A;
        Vec4 CA = C-A;
        float angleA = acos(BA * CA / (BA.length3() * CA.length3()));

        Vec4 AB = A-B;
        Vec4 CB = C-B;
        float angleB = acos(AB * CB / (AB.length3() * CB.length3()));

        Vec4 AC = A-C;
        Vec4 BC = B-C;
        float angleC = acos(AC * BC / (AC.length3() * BC.length3()));

        m_smoothedNormals[triangle.m_vertex[0]] += angleA * N;
        m_smoothedNormals[triangle.m_vertex[1]] += angleB * N;
        m_smoothedNormals[triangle.m_vertex[2]] += angleC * N;
    }

    for (int vi=0; vi<m_smoothedNormals.size(); ++vi)
    {
        Vec4& smoothedN = m_smoothedNormals[vi];
        smoothedN.normalizeIfNotZero();
        smoothedN *= -1.f;
    }

    if (m_vertexNormals.size() > 0) {
        // May need to flip smoothed normals if our assumed winding was wrong (i.e. conflicts with mesh normals).
        // Flip according to majority vote of triangles (normally a global flip will be OK, if winding consistent).
        float flip = 0.0f;

        for (int ti = 0; ti < m_triangles.size(); ++ti) {
            Geometry::Triangle &triangle = m_triangles[ti];

            const Vec4 &NA = m_vertexNormals[triangle.m_vertexNormal[0]];
            const Vec4 &NB = m_vertexNormals[triangle.m_vertexNormal[1]];
            const Vec4 &NC = m_vertexNormals[triangle.m_vertexNormal[2]];

            const Vec4 &NAs = m_smoothedNormals[triangle.m_vertex[0]];
            const Vec4 &NBs = m_smoothedNormals[triangle.m_vertex[1]];
            const Vec4 &NCs = m_smoothedNormals[triangle.m_vertex[2]];

            if ((NA + NB + NC) * (NAs + NBs + NCs) < 0.0f) flip -= 1.f;
        }

        if (flip < 0.f) {
            for (int vi = 0; vi < m_vertices.size(); ++vi) {
                Vec4 &smoothedN = m_smoothedNormals[vi];
                smoothedN *= -1.f;
            }
        }
    }
}

Aabb Geometry::getTriangleAabb(int triIndex) const
{
    Aabb aabb;

    const Geometry::Triangle& t = m_triangles[triIndex];

    Vec4 V0 = m_vertices[t.m_vertex[0]]; V0[3] = 1.f;
    Vec4 V1 = m_vertices[t.m_vertex[1]]; V1[3] = 1.f;
    Vec4 V2 = m_vertices[t.m_vertex[2]]; V2[3] = 1.f;

    aabb.includePoint(V0);
    aabb.includePoint(V1);
    aabb.includePoint(V2);

    return aabb;
}

template<typename V>
class BVH
{
public:

    BVH() : m_built(false) {}

    virtual ~BVH(){}

    /// Deep copy constructor
    BVH(const BVH& rhs);

    // Build whole tree (top down), given all the leaf volumes and values.
    void build( const vector<Aabb>& leafVols, const vector<V>& leafDatas );

    bool isBuilt() const { return m_built; }

    /// Clear tree
    void clear();

    Aabb getRootVol() const;

    int getNumLeaves() const { return m_numLeaves; }
    V getLeafData(int leafIndex) const;

    /// Returns the indices of all leaf nodes which intersect the given box.
    void overlapQuery(const Aabb& space, vector<unsigned int>& leafIndices) const;

    struct Node
    {
        Node() : m_left(INT_MAX), m_right(INT_MAX) {}

        int m_left;  // left node index
        int m_right; // right node index
        Aabb m_volume; // node BV
        int m_leafIndex; // >=0 if and only if this is a leaf
        int m_skipNode;

        inline bool isLeaf() const { return m_leafIndex >= 0; }
    };

    int getNumNodes() const { return m_nodes.size(); }
    const Node& getNode(int nodeIndex) const { return m_nodes[nodeIndex]; }

private:

    // Tree data
    std::vector<Node> m_nodes;
    std::vector<V> m_leafDatas;
    bool m_built;
    unsigned int m_numLeaves;
    int m_highestNode;

    struct TmpNode
    {
        TmpNode() : m_left(NULL), m_right(NULL) {}
        TmpNode* m_left;
        TmpNode* m_right;
        Aabb m_volume;
        int m_leafIndex;
        inline bool isLeaf() const { return m_leafIndex >= 0; }
    };

    // Temporary data structure used during tree build
    TmpNode* m_rootTmpNode;

    void _partitionObjects( const Aabb& nodeVol, const vector<int>& leafIndices, vector<int>& leftSet, vector<int>& rightSet, const vector<Aabb>& leafVols );
    void _buildTreeRecursive( TmpNode** node, const std::vector<int>& leafIndices, const vector<Aabb>& leafVols );
    void _generateNodeIndicesRecursive(TmpNode* tmpnode, int i);
    void _findSkipNodes();
    void _freeTmpNodes(TmpNode* tmpnode);
};


template<typename V>
BVH<V>::BVH(const BVH& rhs)
{
    // Copy the member data across.
    m_nodes = rhs.m_nodes;
    m_leafDatas = rhs.m_leafDatas;
    m_built = rhs.m_built;
    m_numLeaves = rhs.m_numLeaves;
}


template<typename V>
void BVH<V>::build( const vector<Aabb>& leafVols, const std::vector<V>& leafDatas )
{
    if (leafVols.size() != leafDatas.size())
    {
        std::cout << "BVH::build(): supplied leafVols and leafDatas arrays must have same length.";
        return;
    }

    m_numLeaves = leafVols.size();
    m_leafDatas = leafDatas;

    std::vector<int> leafIndices;
    for (unsigned int i=0; i<leafDatas.size(); ++i)
    {
        leafIndices.push_back(int(i));
    }

    if (leafVols.size() > 0)
    {
        _buildTreeRecursive(&m_rootTmpNode, leafIndices, leafVols);
        _generateNodeIndicesRecursive(m_rootTmpNode, 0);
        _findSkipNodes();
        _freeTmpNodes(m_rootTmpNode);
    }

    m_built = true;
}


template<typename V>
void BVH<V>::clear()
{
    m_numLeaves = 0;
    m_nodes.clear();
    m_leafDatas.clear();
    m_built = false;
}


template<typename V>
void BVH<V>::_freeTmpNodes(TmpNode* tmpnode)
{
    if (tmpnode->m_left) _freeTmpNodes(tmpnode->m_left);
    if (tmpnode->m_right) _freeTmpNodes(tmpnode->m_right);
    delete tmpnode;
}


template<typename V>
void BVH<V>::_generateNodeIndicesRecursive(TmpNode* tmpnode, int i)
{
    int numnodes = m_nodes.size();
    if (i >= numnodes)
    {
        int expand = (i+1)*2;
        m_nodes.resize(expand);
    }

    Node& node = m_nodes[i];
    node.m_volume = tmpnode->m_volume;
    node.m_leafIndex = tmpnode->m_leafIndex;
    node.m_left = -1;
    node.m_right = -1;

    // Remap node pointers into an array of 2^(depth+1)-1 indices. Some of these elements will be dummies
    // (flagged by INT_MAX children). Since tree is nearly balanced due to use of median-cut heuristic,
    // this is not too wasteful, and allows for very fast refitting.
    if (tmpnode->m_left)
    {
        m_nodes[i].m_left = 2*i+1;
        _generateNodeIndicesRecursive(tmpnode->m_left, 2*i+1);
    }
    if (tmpnode->m_right)
    {
        m_nodes[i].m_right = 2*i+2;
        _generateNodeIndicesRecursive(tmpnode->m_right, 2*i+2);
    }
}


template<typename V>
void BVH<V>::_findSkipNodes()
{
    // Record in each non-dummy node the index of the previous non-dummy node (-1 flags the first).
    // This allows us to skip dummies when refitting, which is important because at high index there can be large numbers of contiguous dummies.
    int skipNode = -1;
    for (unsigned int n=0; n<m_nodes.size(); n++)
    {
        Node& node = m_nodes[n];
        bool isDummy = (node.m_left==INT_MAX);
        if (!isDummy)
        {
            node.m_skipNode = skipNode;
            skipNode = n;
        }
    }

    // Keep the index of the highest non-dummy node, to start from when refitting.
    m_highestNode = skipNode;
}


template<typename V>
void BVH<V>::_buildTreeRecursive( TmpNode** node, const std::vector<int>& leafIndices, const vector<Aabb>& leafVols )
{
    assert(leafIndices.size() > 0);

    TmpNode* newNode = new TmpNode;
    *node = newNode;

    newNode->m_volume = leafVols[leafIndices[0]];
    for(unsigned int i=1; i<leafIndices.size(); ++i)
    {
        newNode->m_volume.extendBy(leafVols[leafIndices[i]]);
    }

    if (leafIndices.size() == 1)
    {
        newNode->m_left = NULL;
        newNode->m_right = NULL;
        newNode->m_leafIndex = leafIndices[0];
    }
    else
    {
        newNode->m_leafIndex = -1;

        // split the leaves indexed by leafIndices into two (non-empty) sets using the "median-cut" heuristic.
        std::vector<int> leftSet;
        std::vector<int> rightSet;
        _partitionObjects(newNode->m_volume, leafIndices, leftSet, rightSet, leafVols);

        // Recursively construct left and right subtrees
        _buildTreeRecursive(&(newNode->m_left), leftSet, leafVols);
        _buildTreeRecursive(&(newNode->m_right), rightSet, leafVols);
    }
}


struct ProjectedCentroid
{
    float m_position;
    int m_leafIndex;
    friend bool operator<(const ProjectedCentroid& left, const ProjectedCentroid& right);
};

inline bool operator<(const ProjectedCentroid& left, const ProjectedCentroid& right)
{
    return left.m_position < right.m_position;
}


template<typename V>
void BVH<V>::_partitionObjects( const Aabb& nodeVol, const std::vector<int>& leafIndices,
                                std::vector<int>& leftSet, std::vector<int>& rightSet, const std::vector<Aabb>& leafVols )
{
    int numLeaves = leafIndices.size();
    // Choose partition axis as axis of maximum extent of nodeVol.
    int partitionAxis = nodeVol.majorAxis();
    // Partition into two sets according to projected (on axis) centroid location w.r.t. median centroid location.
    std::vector<ProjectedCentroid> projectedCentroids;
    for (int i=0; i<numLeaves; ++i)
    {
        const Aabb& leafVol = leafVols[leafIndices[i]];
        Vec4 leafVolCentroid = leafVol.center();

        ProjectedCentroid pc;
        pc.m_position = leafVolCentroid[partitionAxis];
        pc.m_leafIndex = leafIndices[i];
        projectedCentroids.push_back(pc);
    }

    std::sort(projectedCentroids.begin(), projectedCentroids.end());

    unsigned int median = numLeaves/2;

    // Since numLeaves>1 by construction, 0<median<numLeaves, so leftSet and rightSet are non-empty (and disjoint).
    for (unsigned int i=0; i<median; ++i)
    {
        leftSet.push_back(projectedCentroids[i].m_leafIndex);
    }
    for (unsigned int i=median; i<projectedCentroids.size(); ++i)
    {
        rightSet.push_back(projectedCentroids[i].m_leafIndex);
    }
}

template<typename V>
Aabb BVH<V>::getRootVol() const
{
    if (!m_nodes.size())
    {
        Aabb box;
        return box;
    }
    const Node& root = m_nodes[0];
    return root.m_volume;
}


template<typename V>
V BVH<V>::getLeafData(int leafIndex) const
{
    return m_leafDatas[leafIndex];
}

template<typename V>
inline void BVH<V>::overlapQuery(const Aabb& space, std::vector<unsigned int>& leafIndices) const
{
    std::stack<int> S;
    S.push(0);
    while ( S.size() )
    {
        int nodeIndex = S.top();
        S.pop();
        const Node& node = m_nodes[nodeIndex];
        if ( !node.m_volume.intersects(space) ) continue;
        if (node.isLeaf())
        {
            unsigned int leafIndex = static_cast<unsigned int>(node.m_leafIndex);
            leafIndices.push_back(leafIndex);
        }
        else
        {
            assert(node.m_left >= 0);
            assert(node.m_right >= 0);
            S.push(node.m_left);
            S.push(node.m_right);
        }
    }
}


template<typename T> inline T flt_epsilon() { return 0.0; }
template<> inline float flt_epsilon<float>() { return FLT_EPSILON; }
template<> inline double flt_epsilon<double>() { return DBL_EPSILON; }

struct SDFComputer
{
    SDFComputer(const Geometry& geometry) : m_geometry(geometry)
    {
        std::cout << "Building BVH ..." << std::endl;
        vector<Aabb> faceAabbs;
        for (unsigned int ti=0; ti<geometry.m_triangles.size(); ++ti)
        {
            const Geometry::Triangle& tri = geometry.m_triangles[ti];
            Aabb faceAabb;
            faceAabb.extendBy(geometry.m_vertices[tri.m_vertex[0]]);
            faceAabb.extendBy(geometry.m_vertices[tri.m_vertex[1]]);
            faceAabb.extendBy(geometry.m_vertices[tri.m_vertex[2]]);
            faceAabbs.push_back(faceAabb);

            m_leafIndexToFaceIndex.push_back(ti);
        }
        m_bvh.build(faceAabbs, m_leafIndexToFaceIndex);
    }

    template<typename T> int sgn(T val) const {
        return (T(0) < val) - (val < T(0));
    }

    inline double getSDF(const Vec4 &wsP) const
    {
        Vec4 closestPoint, closestPointNormal;
        getClosestPointOnMesh(wsP, closestPoint, closestPointNormal);
        Vec4 delta = wsP - closestPoint;
        return sgn(delta ^ closestPointNormal) * delta.length3();
    }

    private:

    inline bool getClosestPointOnMesh(const Vec4 &wsP, Vec4 &closestPoint, Vec4 &closestPointNormal) const
    {
        const float startBoxSize = 3.0e-2 * m_bvh.getRootVol().size();
        const float growthRate = 2.0;
        const int maxIters = 256;
        // Make a test cube of the specified size centered on the test point wsP
        Aabb box;
        box.extendBy(wsP);
        box.m_maxima += Vec4(0.5*startBoxSize);
        box.m_minima -= Vec4(0.5*startBoxSize);
        // Expand test cube exponentially until the embedded sphere hits the mesh
        int nIter = 0;
        bool foundPoint = false;
        while ( !foundPoint && nIter++<maxIters )
        {
            std::vector<unsigned int> leafIndices;
            m_bvh.overlapQuery(box, leafIndices);
            if (leafIndices.size()>0)
            {
                // Reject points which aren't within the sphere contained within the test cube,
                // otherwise we don't necessarily get the closest point.
                float radius = 0.5 * (box.m_maxima[0] - box.m_minima[0]);
                float minDistSqr = radius*radius;
                for (unsigned int leaf=0; leaf<leafIndices.size(); ++leaf)
                {
                    unsigned int faceIndex = m_leafIndexToFaceIndex[leafIndices[leaf]];
                    const Geometry::Triangle& face = m_geometry.m_triangles[faceIndex];
                    const Vec4& vA = m_geometry.m_vertices[face.m_vertex[0]];
                    const Vec4& vB = m_geometry.m_vertices[face.m_vertex[1]];
                    const Vec4& vC = m_geometry.m_vertices[face.m_vertex[2]];
                    float w[3];
                    Vec4 P = getClosestPointOnTriangle(wsP, vA, vB, vC, w);
                    Vec4 triToWsP = wsP - P;
                    float dSqr = triToWsP.length3Sqr();
                    if (dSqr < minDistSqr)
                    {
                        foundPoint = true;
                        minDistSqr = dSqr;
                        closestPoint = P;
                        const Vec4& nA = m_geometry.m_smoothedNormals[face.m_vertex[0]];
                        const Vec4& nB = m_geometry.m_smoothedNormals[face.m_vertex[1]];
                        const Vec4& nC = m_geometry.m_smoothedNormals[face.m_vertex[2]];
                        closestPointNormal = w[0]*nA + w[1]*nB + w[2]*nC;
                    }
                }
            }
            Vec4 expand(growthRate*(box.m_maxima-box.m_minima).length3());
            box.m_maxima += expand;
            box.m_minima -= expand;
        }
        if (nIter>=maxIters)
        {
            std::cout << "Failed to find a closest point. Increase iteration count or growth rate." << std::endl;
            assert(0);
        }
        return true;
    }

    private:

    static inline void getProjectedBarycoords(const Vec4& p,
                                              const Vec4& X0, const Vec4& X1, const Vec4& X2,
                                              float* w)
    {
        Vec4 X10 = X1 - X0;
        Vec4 X20 = X2 - X0;
        float A = X10 ^ X10;
        float B = X20 ^ X20;
        float C = X10 ^ X20;
        float det = A*B - C*C;
        if (fabs(det)<flt_epsilon<float>())
        {
            // Degenerate triangle
            w[0] = w[1] = w[2] = 1.0/3.0;
            return;
        }
        float invDet = 1.0/det;
        Vec4 XX0 = p - X0;
        float y0 = X10 ^ XX0;
        float y1 = X20 ^ XX0;
        w[1] = ( B*y0 - C*y1) * invDet;
        w[2] = (-C*y0 + A*y1) * invDet;
        w[0] = 1.0 - w[1] - w[2];
    }

    static Vec4 closestPointOnLineSegment( const Vec4& X, const Vec4& a, const Vec4& b )
    {
        Vec4 D = X - a;
        Vec4 BA = b - a;
        float l = (D^BA) / ((BA^BA) + flt_epsilon<float>());
        if (l>1.0) l = 1.0;
        if (l<0.0) l = 0.0;
        Vec4 Xp = a + l*BA;
        return Xp;
    }

    static inline float pointToLineSegment( const Vec4& X, const Vec4& a, const Vec4& b, Vec4& Xp ) {
        Xp = closestPointOnLineSegment(X, a, b);
        Vec4 d = X - Xp;
        return d.length3();
    }

    /// Get the closest point on the triangle with vertices A, B, C to the point P (NB, this can be either on the triangle, or on an edge, or on a vertex).
    static inline Vec4 getClosestPointOnTriangle(const Vec4& P, const Vec4& A, const Vec4& B, const Vec4& C, float w[3])
    {
        getProjectedBarycoords(P, A, B, C, w);
        // If the projected point X lies inside the triangle, then the closest point is X
        const float lo = -flt_epsilon<float>();
        const float hi = 1.0 - lo;
        if (w[0]>=lo && w[0]<=hi &&
            w[1]>=lo && w[1]<=hi &&
            w[2]>=lo && w[2]<=hi )
        {
            Vec4 X = w[0]*A + w[1]*B + w[2]*C;
            return X;
        }
        // Otherwise, the closest point is the projection of P onto the closest edge
        Vec4 X01, X12, X20;
        float dX01 = pointToLineSegment( P, A, B, X01 );
        float dX12 = pointToLineSegment( P, B, C, X12 );
        float dX20 = pointToLineSegment( P, C, A, X20 );
        if (dX01<dX20)
        {
            if (dX12<dX01) return X12;
            return X01;
        }
        else
        {
            if (dX20<dX12) return X20;
            return X12;
        }
    }

private:

    const Geometry& m_geometry;
    BVH<unsigned int> m_bvh;
    std::vector<unsigned int> m_leafIndexToFaceIndex;

};

class timerutil {
public:
#ifdef _WIN32
    typedef DWORD time_t;

  timerutil() { ::timeBeginPeriod(1); }
  ~timerutil() { ::timeEndPeriod(1); }

  void start() { t_[0] = ::timeGetTime(); }
  void end() { t_[1] = ::timeGetTime(); }

  time_t sec() { return (time_t)((t_[1] - t_[0]) / 1000); }
  time_t msec() { return (time_t)((t_[1] - t_[0])); }
  time_t usec() { return (time_t)((t_[1] - t_[0]) * 1000); }
  time_t current() { return ::timeGetTime(); }

#else
#if defined(__unix__) || defined(__APPLE__)
    typedef unsigned long int time_t;

    void start() { gettimeofday(tv + 0, &tz); }
    void end() { gettimeofday(tv + 1, &tz); }

    time_t sec() { return static_cast<time_t>(tv[1].tv_sec - tv[0].tv_sec); }
    time_t msec() {
        return this->sec() * 1000 +
               static_cast<time_t>((tv[1].tv_usec - tv[0].tv_usec) / 1000);
    }
    time_t usec() {
        return this->sec() * 1000000 +
               static_cast<time_t>(tv[1].tv_usec - tv[0].tv_usec);
    }
    time_t current() {
        struct timeval t;
        gettimeofday(&t, NULL);
        return static_cast<time_t>(t.tv_sec * 1000 + t.tv_usec);
    }

#else  // C timer
    // using namespace std;
  typedef clock_t time_t;

  void start() { t_[0] = clock(); }
  void end() { t_[1] = clock(); }

  time_t sec() { return (time_t)((t_[1] - t_[0]) / CLOCKS_PER_SEC); }
  time_t msec() { return (time_t)((t_[1] - t_[0]) * 1000 / CLOCKS_PER_SEC); }
  time_t usec() { return (time_t)((t_[1] - t_[0]) * 1000000 / CLOCKS_PER_SEC); }
  time_t current() { return (time_t)clock(); }

#endif
#endif

private:
#ifdef _WIN32
    DWORD t_[2];
#else
#if defined(__unix__) || defined(__APPLE__)
    struct timeval tv[2];
    struct timezone tz;
#else
    time_t t_[2];
#endif
#endif
};

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

/////////////////////////////////////////////////////////////////////////////////////////
// Octree
/////////////////////////////////////////////////////////////////////////////////////////


/// Mimics a 64 bit number by two 32 bit numbers, for glsl
struct uint64_glsl
{
    uint32_t m_lo;
    uint32_t m_hi;
};

// A 'small' (i.e. <=32 bit) roll left
static void uint64_glsl_rollleft_small(unsigned char shift, uint64_glsl& io)
{
    io.m_hi <<= shift;
    uint32_t lo = io.m_lo;
    io.m_lo = lo << shift;
    uint32_t top_lo = lo >> (32-shift); // extract the top 'shift' bits of lo
    io.m_hi |= top_lo;                 // copy them into the low 'shift' bits of io->m_hi
}

// A 'big' (i.e. >=32 bit) roll left
static void uint64_glsl_rollleft_big(unsigned char shift, uint64_glsl& io)
{
    io.m_hi = io.m_lo << (shift-32);
    io.m_lo = 0;
}

// A 'small' (i.e. <=32 bit) roll right
static void uint64_rollright_small(unsigned char shift, uint64_glsl& io)
{
    io.m_lo >>= shift;
    uint32_t hi = io.m_hi;
    io.m_hi = hi >> shift;
    unsigned char inv_shift = 32-shift;
    uint32_t bottom_hi = (hi<<inv_shift) >> inv_shift; // isolate the bottom 'shift' bits of hi
    io.m_lo |= bottom_hi;                              // copy them into the high 'shift' bits of io->m_lo
}

// A 'big' (i.e. >=32 bit) roll right
static void uint64_rollright_big(unsigned char shift, uint64_glsl& io)
{
    io.m_lo = io.m_hi >> (shift-32);
    io.m_hi = 0;
}




class Octree
{
public:

    Octree(const Geometry& geometry, const SDFComputer& sdf, const float sdfAccuracy=1.0e-1, const int maxDepth=5);

    struct NodeSDF
    {
        double sdf[8]; // SDFs at node corners in standard binary order
    };

    typedef pair<uint32_t, NodeSDF> LEAF;

    // return index of code in leaves, if it exists, otherwise -1
    static int binarySearch(uint32_t code, const vector<LEAF>& leaves)
    {
        int left = 0;
        int right = leaves.size()-1;
        while (left <= right)
        {
            int middle = (left + right)/2;
            const LEAF& L = leaves[middle];
            if      (L.first == code) return middle;
            else if (L.first > code) right = middle-1;
            else                      left = middle+1;
        }
        return -1;
    }

    // Reference code for how to sample the SDF via the node and SDF lists
    double sampleSDF(const Vec4& wsP, const vector<LEAF>& leaves) const
    {
        Vec4 lsP = (wsP - m_origin) / m_edge;
        char depth=0;
        while (depth<=m_maxDepth)
        {
            uint32_t code = genNodeCode(lsP, depth);

            // if code exists in the sorted list of leaves, return its tri-linearly interpolated value.
            int leaf = binarySearch(code, leaves);
            if (leaf >= 0)
            {
                const NodeSDF& nSDF = leaves[leaf].second;
                Aabb leafCube = nodeBox(code);

                Vec4 f2 = (lsP - leafCube.m_minima) * invNodeSize(nodeDepth(code));
                Vec4 f1 = Vec4(1.0) - f2;

                double mmm = nSDF.sdf[0]; // x=f1, y=f1, z=f1
                double pmm = nSDF.sdf[1]; // x=f2, y=f1, z=f1
                double mpm = nSDF.sdf[2]; // x=f1, y=f2, z=f1
                double ppm = nSDF.sdf[3]; // x=f2, y=f2, z=f1
                double mmp = nSDF.sdf[4]; // x=f1, y=f1, z=f2
                double pmp = nSDF.sdf[5]; // x=f2, y=f1, z=f2
                double mpp = nSDF.sdf[6]; // x=f1, y=f2, z=f2
                double ppp = nSDF.sdf[7]; // x=f2, y=f2, z=f2

                // Do the lerp (note that by doing the clamp above we ensured that the indices supplied to node.block_value() are in the range [-1, B])
                return ( f1.x() * (f1.y()*(f1.z()*mmm + f2.z()*mmp)  +
                                   f2.y()*(f1.z()*mpm + f2.z()*mpp)) +
                         f2.x() * (f1.y()*(f1.z()*pmm + f2.z()*pmp)  +
                                   f2.y()*(f1.z()*ppm + f2.z()*ppp)) );
            }
            depth++;
        }
        assert(0); // shouldn't reach here
        return 0.0;
    }

    static bool compareLEAF(const LEAF& i, const LEAF& j)
    {
        return i.first < j.first;
    }

    // returns list of leaf nodes (code and corner SDFs) sorted by code
    void getLeaves(vector<LEAF>& leaves) const
    {
        auto iter = MAP.begin();
        for (; iter != MAP.end(); ++iter)
        {
            uint64_t key = iter->first;
            const NodeSDF& sdf = iter->second;
            leaves.push_back( LEAF(key, sdf) );
        }
        sort(leaves.begin(), leaves.end(), compareLEAF);
    }

    Vec4 getOrigin() const { return m_origin; }
    double getEdge() const { return m_edge; }

private:

        const Geometry& m_geometry;
        const SDFComputer& m_sdf;
        Vec4 m_origin; // World space origin of octree cube (i.e. lower left corner)
        double m_edge; // World space edge length of octree cube
        int m_maxDepth;

        // Map from the existing leaf nodes, to their SDF values
        std::unordered_map<uint32_t, NodeSDF> MAP;

public:

        static inline uint32_t part1By2(uint32_t x)
        {
            x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
            x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
            x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
            x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
            x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
            return x;
        }

        static inline uint32_t mortonEncode(uint32_t x, uint32_t y, uint32_t z)
        {
            return (part1By2(z)<<2) + (part1By2(y)<<1) + part1By2(x);
        }

        static inline uint32_t compact1By2(uint32_t x)
        {
            x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
            x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
            x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
            x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
            x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
            return x;
        }

        static inline void mortonDecode(uint32_t k, uint32_t& x, uint32_t& y, uint32_t& z)
        {
            x = compact1By2(k >> 0);
            y = compact1By2(k >> 1);
            z = compact1By2(k >> 2);
        }

        static inline uint32_t encodeMorton(const Vec4& P)
        {
            uint32_t Px = ((uint32_t)(P.x()*(1UL<<9)));
            uint32_t Py = ((uint32_t)(P.y()*(1UL<<9)));
            uint32_t Pz = ((uint32_t)(P.z()*(1UL<<9)));
            return mortonEncode(Px, Py, Pz);
        }

        static inline Vec4 decodeMorton(uint32_t key, bool debug=false)
        {
            static const double norm = 1.0/double(1UL<<9);
            uint32_t x, y, z;
            mortonDecode(key, x, y, z);
            Vec4 lsP = Vec4(double(x), double(y), double(z))*norm;
            return lsP;
        }

        static inline Vec4 nodeCorner(uint32_t nodeCode)
        {
            uint32_t depth = nodeDepth(nodeCode); // Extract depth bits
            nodeCode &= ~(depth<<27UL);           // Zero depth bits
            char d3 = 3*depth;                    // Construct the original Morton key
            uint32_t key = nodeCode << (27UL-d3);
            Vec4 tmp = decodeMorton(key);
            return decodeMorton(key);
        }

        static inline uint32_t genNodeCode(const Vec4& lsP, unsigned char depth)
        {
            assert(depth<=9);
            uint32_t key = encodeMorton(lsP);
            uint32_t d3 = 3*depth;  // At depth d, a key has 3*d bits  (d=0 is the root, d=1 is the first 8 children, etc.)
            key >>= (27UL-d3);      // Strip all but those 3*d bits from the key.
            // We store the depth of the node in the top 5 bits of the 32 bit key.
            // The lower 27 bits contain the location code (up to depth 9, since 3*9=27), only the first (3*depth) of which contain the actual code (the rest are zero).
            // (Note, the code of the root node is exactly 0).
            key |= ((uint32_t)depth<<27UL);
            return key;
        }

        static inline void nodeCodeToVoxelIndex(uint32_t nodeCode, uint32_t& x, uint32_t& y, uint32_t& z)
        {
            uint32_t depth = nodeDepth(nodeCode);
            nodeCode &= ~(depth<<27UL); // Zero depth bits
            mortonDecode(nodeCode, x, y, z);
        }

        static inline unsigned char nodeDepth(uint32_t nodeCode)
        {
            return static_cast<unsigned char>(nodeCode>>27UL);
        }

        static inline double nodeSize(char depth)
        {
            uint32_t res = 1UL<<depth;
            return 1.0/double(res);
        }

        static inline double invNodeSize(char depth)
        {
            uint32_t res = 1UL<<depth;
            return double(res);
        }

        static inline Aabb nodeBox(uint32_t nodeCode)
        {
            // Given the location code of a node, determine the floating point bbox of the cube corresponding to the node:
            Vec4 lsP = nodeCorner(nodeCode);            // extract the floating point coordinates of the center of the node
            Vec4 extent(nodeSize(nodeDepth(nodeCode))); // determine the node extent from the depth,
            Aabb b; b.m_minima = lsP; b.m_maxima = lsP+extent;  // and thus construct the node bounding box:
            return b;
        }

        static inline uint64_t childCode(uint32_t parentCode, char childIndex)
        {
            uint32_t parentDepth = nodeDepth(parentCode);            // Extract depth bits
            assert(parentDepth<9);                                   // Maximum depth nodes cannot have children (must be leaves)
            uint32_t childCode = parentCode & ~(parentDepth<<27UL);  // Zero depth bits
            childCode = (childCode<<3) + childIndex;
            childCode |= ((parentDepth+1)<<27UL);                    // Set depth bits to child's depth
            return childCode;
        }
};

Octree::Octree(const Geometry& geometry, const SDFComputer& sdf, const float sdfAccuracy, const int maxDepth) :

        m_geometry(geometry), m_sdf(sdf), m_maxDepth(maxDepth)

{
    Aabb aabb = geometry.computeAabb();

    // Compute dimensions of octree cube which bounds mesh
    m_origin = aabb.m_minima;
    Vec4 extent = aabb.m_maxima - aabb.m_minima;
    extent.maxIndex(m_edge);

    std::queue<uint32_t> Q;
    Q.push(0);

    double processedFraction = 0;
    int lastReport = -1;

    while (!Q.empty())
    {
        int processedPercent = int(100.0*processedFraction);
        if (processedPercent > lastReport)
        {
            std::cout << processedPercent << "% done .." << std::endl;
            lastReport = processedPercent;
        }

        uint32_t nodeCode = Q.front();
        Q.pop();

        unsigned char _nodeDepth = nodeDepth(nodeCode);

        // Take SDF samples at block corners, to determine whether to bust block.
        Aabb nodeBoxL = nodeBox(nodeCode);

        Aabb nodeBoxW;
        nodeBoxW.m_minima = m_origin + m_edge*(nodeBoxL.m_minima);
        nodeBoxW.m_maxima = m_origin + m_edge*(nodeBoxL.m_maxima);

        float sdfC = m_sdf.getSDF(nodeBoxW.center());

        Vec4 corners[8];                                                                        // zyx
        corners[0] = Vec4(nodeBoxW.m_minima.x(), nodeBoxW.m_minima.y(), nodeBoxW.m_minima.z()); // 000
        corners[1] = Vec4(nodeBoxW.m_maxima.x(), nodeBoxW.m_minima.y(), nodeBoxW.m_minima.z()); // 001
        corners[2] = Vec4(nodeBoxW.m_minima.x(), nodeBoxW.m_maxima.y(), nodeBoxW.m_minima.z()); // 010
        corners[3] = Vec4(nodeBoxW.m_maxima.x(), nodeBoxW.m_maxima.y(), nodeBoxW.m_minima.z()); // 011
        corners[4] = Vec4(nodeBoxW.m_minima.x(), nodeBoxW.m_minima.y(), nodeBoxW.m_maxima.z()); // 100
        corners[5] = Vec4(nodeBoxW.m_maxima.x(), nodeBoxW.m_minima.y(), nodeBoxW.m_maxima.z()); // 101
        corners[6] = Vec4(nodeBoxW.m_minima.x(), nodeBoxW.m_maxima.y(), nodeBoxW.m_maxima.z()); // 110
        corners[7] = Vec4(nodeBoxW.m_maxima.x(), nodeBoxW.m_maxima.y(), nodeBoxW.m_maxima.z()); // 111

        // Always bust into the first 8 children at least
        bool shouldBust = _nodeDepth<=1;
        shouldBust = true; // !!! For now always generate fully busted octree, i.e. all leaves are at max depth

        // And also bust octree node if any corner's SDF differs from the center SDF by more than a tolerance distance
        NodeSDF nSDF;
        for (int c=0; c<8; ++c)
        {
            const Vec4& P = corners[c];
            float value = m_sdf.getSDF(P);
            nSDF.sdf[c] = value;
            shouldBust |= fabs(sdfC - value) > sdfAccuracy*m_edge;
        }

        if (shouldBust && _nodeDepth<maxDepth)
        {
            for (char c=0; c<8; ++c)
            {
                uint32_t _childCode = childCode(nodeCode, c);
                Q.push(_childCode);
            }
            continue;
        }

        // Create a leaf node (if we either reached max depth, or if none of the criteria for busting further were met):
        processedFraction += nodeBoxL.volume();
        MAP[nodeCode] = nSDF;
    }
}

class Grid {
public:

    Grid(const Geometry &geometry, const SDFComputer &sdf, const int resolution) :

            m_geometry(geometry), m_sdf(sdf)

    {
        Aabb aabb = geometry.computeAabb();

        m_origin = aabb.m_minima;
        Vec4 extent = aabb.m_maxima - aabb.m_minima;

        double max_edge;
        int max_cardinal = extent.maxIndex(max_edge);
        m_voxelWidth = max_edge / double(resolution);
        m_res[max_cardinal] = resolution;
        m_res[(max_cardinal+1)%3] = ceil(extent[(max_cardinal+1)%3] / m_voxelWidth);
        m_res[(max_cardinal+2)%3] = ceil(extent[(max_cardinal+2)%3] / m_voxelWidth);

        m_numValues = (m_res[0]) * (m_res[1]) * (m_res[2]);
        m_data = new double[m_numValues];

        // sample SDFs
        for (size_t k=0; k<m_res[2]; ++k)
        {
            std::cout << "Sampling SDF on grid, " << 100.0*double(k)/double(m_res[0]) << "% done" << std::endl;
            double z = m_origin[2] + (double(k)+0.5)*m_voxelWidth;

            for (size_t j=0; j<m_res[1]; ++j)
            {
                double y = m_origin[1] + (double(j)+0.5)*m_voxelWidth;
                for (size_t i=0; i<m_res[0]; ++i)
                {
                    double x = m_origin[0] + (double(i)+0.5)*m_voxelWidth;
                    int data_index = latticeToIndex(i, j, k);
                    Vec4 wsP(x, y, z);
                    m_data[data_index] = m_sdf.getSDF(wsP);
                }
            }
        }
    }

    string getDataAsPython() const
    {
        string PYTHON;

        PYTHON += "grid_origin = [" + to_string(m_origin.x()) + ", " + to_string(m_origin.y()) + ", " + to_string(m_origin.z()) + "]";
        PYTHON += "\nvoxel_resolution = [" + to_string(m_res[2]) + ", " + to_string(m_res[1]) + ", " + to_string(m_res[0]) + "]";
        PYTHON += "\nvoxel_size = " + to_string(m_voxelWidth);

        PYTHON += "\nsdfs = [";
        for (size_t n=0; n<m_numValues; ++n)
        {
            if (n>0) PYTHON += ", ";
            PYTHON += to_string(m_data[n]);
        }
        PYTHON += "]";

        return PYTHON;
    }

    string getPackedDataAsPython() const
    {
        // identify max absolute SDF value.
        double sdf_absmax = 0.0;
        for (size_t n=0; n<m_numValues; ++n)
        {
            double s = m_data[n];
            if (fabs(s) > sdf_absmax) sdf_absmax = s;
        }

        std::vector<unsigned char> char_data;
        for (size_t n=0; n<m_numValues; ++n)
        {
            double s = m_data[n];

            // divide each SDF by absolute max, putting each in range [-1,1].
            // multiple each by 127.5, and add 127.5, putting in range [0, 255].
            s *= 127.5/sdf_absmax;
            s += 127.5;
            unsigned char c = max((unsigned char)0, min((unsigned char)255, static_cast<unsigned char>(round(s))));
            char_data.push_back(c);
        }

        // proceed by uint, and pack 4 SDFs into each uint.
        std::vector<unsigned int> packed_data;
        int n = 0;
        while (n < m_numValues)
        {
            unsigned int byte0 = (n<m_numValues) ? (unsigned int) char_data[n] : 0; n++;
            unsigned int byte1 = (n<m_numValues) ? (unsigned int) char_data[n] : 0; n++;
            unsigned int byte2 = (n<m_numValues) ? (unsigned int) char_data[n] : 0; n++;
            unsigned int byte3 = (n<m_numValues) ? (unsigned int) char_data[n] : 0; n++;
            unsigned int pack = byte3 | (byte2 << 8) | (byte1 << 16) | (byte0 << 24);
            packed_data.push_back(pack);
        }

        string PYTHON;

        PYTHON += "grid_origin = [" + to_string(m_origin.x()) + ", " + to_string(m_origin.y()) + ", " + to_string(m_origin.z()) + "]\n";
        PYTHON += "\nvoxel_resolution = [" + to_string(m_res[2]) + ", " + to_string(m_res[1]) + ", " + to_string(m_res[0]) + "]\n";
        PYTHON += "\nvoxel_size = " + to_string(m_voxelWidth);
        PYTHON += "\nsdf_absmax = " + to_string(sdf_absmax);

        PYTHON += "\npacked_sdfs = [";
        for (size_t n=0; n<packed_data.size(); ++n)
        {
            if (n>0) PYTHON += ", ";
            PYTHON += to_string(packed_data[n]);
        }
        PYTHON += "]";

        // Check
        for (size_t sdf_index=0; sdf_index<m_numValues; ++sdf_index)
        {
            double s = m_data[sdf_index];

            unsigned int packed_sdfs_index = sdf_index/4;
            unsigned int byte = sdf_index - packed_sdfs_index*4;
            unsigned int packed_sdf = packed_data[packed_sdfs_index];
            unsigned int sdf_int = (packed_sdf >> 8u*(3u-byte)) & 0xFF;

            double s_reconstruct = sdf_absmax * (static_cast<double>(sdf_int) - 127.5) / 127.5;
            //std::cout << "s, s_reconstruct: " << to_string(s) << ", " << to_string(s_reconstruct) << std::endl;
        }

        return PYTHON;
    }

    template<typename T>
    T clamp(T n, T lower, T upper) { return std::max(lower, std::min(n, upper)); }

    // given world space point P, return lattice index of grid point at LL of voxel containing P
    void worldToVoxel(const Vec4& wsP, int& x, int& y, int& z)
    {
        Vec4 vsP = (wsP - m_origin) / m_voxelWidth;
        x = clamp(int(vsP[0]), 0, m_res[0]);
        y = clamp(int(vsP[1]), 0, m_res[1]);
        z = clamp(int(vsP[2]), 0, m_res[2]);
    }

    int latticeToIndex(int x, int y, int z)
    {
        return z + m_res[2]*(y + m_res[1]*x);
    }

private:

    const Geometry& m_geometry;
    const SDFComputer& m_sdf;

    Vec4 m_origin;
    double m_voxelWidth;
    int m_res[3]; // voxel counts along grid edges

    // voxel data at all voxel corners
    int m_numValues; // (m_res[0]+1) * (m_res[1]+1) * (m_res[2]+1)
    double *m_data; // m_numValues doubles

};

/////////////////////////////////////////////////////////////////////////////////////////
// main
/////////////////////////////////////////////////////////////////////////////////////////


static void doIt()
{
    std::string inputfile = "../resources/bunny.obj";
    //std::string inputfile = "../resources/sphere.obj";

    std::cout << "Loading " << inputfile << std::endl;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    timerutil t;
    t.start();
    std::string err;
    bool triangulate = true;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, inputfile.c_str(),
                                NULL, triangulate);
    t.end();
    printf("Parsing time: %lu [msecs]\n", t.msec());

    std::cout << "Constructing geometry ... " << std::endl;
    Geometry geometry;

    size_t numVertices = attrib.vertices.size()/3;
    for (size_t vi=0; vi<numVertices; vi++)
    {
        Vec4 vertex(&attrib.vertices[3*vi]);
        geometry.m_vertices.push_back(vertex);
    }

    // Construct topology
    for (size_t i = 0; i < shapes.size(); i++)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++)
        {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];
            for (size_t v=2; v<fnum; v++)
            {
                Geometry::Triangle T;

                tinyobj::index_t idxA = shapes[i].mesh.indices[index_offset + v];
                tinyobj::index_t idxB = shapes[i].mesh.indices[index_offset + v - 1];
                tinyobj::index_t idxC = shapes[i].mesh.indices[index_offset + v - 2];

                T.m_vertex[0] = idxA.vertex_index;
                T.m_vertex[1] = idxB.vertex_index;
                T.m_vertex[2] = idxC.vertex_index;

                T.m_vertexNormal[0] = idxA.normal_index;
                T.m_vertexNormal[1] = idxB.normal_index;
                T.m_vertexNormal[2] = idxC.normal_index;

                T.m_uvs[0] = idxA.texcoord_index;
                T.m_uvs[1] = idxB.texcoord_index;
                T.m_uvs[2] = idxC.texcoord_index;

                geometry.m_triangles.push_back(T);
            }
            index_offset += fnum;
        }
    }

    std::cout << "Computing smoothed normals ..." << std::endl;
    geometry.computeSmoothedNormals();

    std::cout << "\t num vertices:  " << geometry.m_vertices.size() << std::endl;
    std::cout << "\t num triangles: " << geometry.m_triangles.size() << std::endl;

    SDFComputer SDF(geometry);

    // Build adaptive SDF octree
    /*
    const float sdfVariance=0.0;
    const int maxDepth=2;
    Octree octree(geometry, SDF, sdfVariance, maxDepth);

    vector<Octree::LEAF> leaves;
    octree.getLeaves(leaves);
     */

    Grid grid(geometry, SDF, 32);

    std::ofstream out("../sdf.txt");

    std::cout  << "\n\n########################## Python code #################################\n" << std::endl;
    string data = grid.getDataAsPython();
    std::cout << data << std::endl;
    out << data;
    std::cout  << "\n########################## Python code  end #################################\n" << std::endl;


    /*
    string PYTHON;

    // write node codes array (successive lo, hi bits of sorted Morton codes)
    std::cout << "num leaf nodes = " << leaves.size() << std::endl;

    std::cout << "\n\n########################## Python code #################################\n" << std::endl;

    Vec4 O = octree.getOrigin();
    PYTHON += "octree_origin = [" + to_string(O.x()) + ", " + to_string(O.y()) + ", " + to_string(O.z()) + "]\n";
    PYTHON += "octree_edge = " + to_string(octree.getEdge());

    unsigned char maxLeafDepth = 0;
    PYTHON += "\nmorton_codes = [";
    for (size_t l=0; l<leaves.size(); ++l)
    {
        uint32_t code = leaves[l].first;
        unsigned char depth = Octree::nodeDepth(code);
        if (depth>maxLeafDepth) maxLeafDepth = depth;
        if (l>0) PYTHON += ", ";
        PYTHON += to_string(code);
    }
    PYTHON += "]";

    PYTHON += "\nsdfs = [";
    for (size_t l=0; l<leaves.size(); ++l)
    {
        uint64_t code = leaves[l].first;
        const Octree::NodeSDF &sdf = leaves[l].second;
        if (l>0) PYTHON += ", ";
        for (size_t n=0; n<8; ++n)
        {
            if (n>0) PYTHON += ", ";
            PYTHON += to_string(sdf.sdf[n]);
        }
    }
    PYTHON += "]";

    PYTHON += "\nmaxLeafDepth = " + to_string(maxLeafDepth);

    std:cout << PYTHON << std::endl;
    std::cout << "\n########################## Python code  end #################################\n" << std::endl;

     */


    /*
        for (size_t l=0; l<leaves.size(); ++l)
        {
            uint32_t code = leaves[l].first;
            std::cout << "\nuint32_t:  " << code << std::endl;

            const Octree::NodeSDF& sdf = leaves[l].second;
            std::cout << "sdf:  " << sdf.sdf[0] << ", " << sdf.sdf[1] << ", " << sdf.sdf[2] << ", " << sdf.sdf[3] << ", "
                                  << sdf.sdf[4] << ", " << sdf.sdf[5] << ", " << sdf.sdf[6] << ", " << sdf.sdf[7] << std::endl;

            Vec4 lsP = Octree::nodeCorner(code);
            std::cout << "corner lsP: " << lsP.x() << ", " << lsP.y() << ", " << lsP.z() << std::endl;

            unsigned char depth = Octree::nodeDepth(code);
            std::cout << "depth:  " << (int)depth << std::endl;

            uint32_t x, y, z;
            Octree::nodeCodeToVoxelIndex(code, x, y, z);
            std::cout << "voxel (" << x << ", " << y << ", " << z << ")" << std::endl;

            uint32_t validation_code = Octree::genNodeCode(lsP, depth);
            std::cout << "original code:   " << std::bitset<32>(code) << std::endl;
            std::cout << "validation code: " << std::bitset<32>(validation_code) << std::endl;

            assert( Octree::binarySearch(code, leaves)>=0 );
        }
        */

}

int main()
{
    doIt();
    std::cout << "Done." << std::endl;
    return 0;
}
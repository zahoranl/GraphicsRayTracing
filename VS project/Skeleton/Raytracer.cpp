#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		 
#include <GL/freeglut.h>	
#endif
//const unsigned int windowWidth = 1920, windowHeight = 1080;
const unsigned int windowWidth = 600, windowHeight = 600;
int majorVersion = 3, minorVersion = 3;
const float Epsilon = 10e-4;
class Object;
Object* objects[50000];
int objdb = 0;
#pragma region MATH STRUCTS
struct vec3 {
	float x,y, z;
	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator/(const vec3& v) const {
		return vec3(x / v.x, y / v.y, z / v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }
	operator float*() { return &x; }
};
struct vec4 {
	float v[4];
	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}
};
float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}
vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}
struct mat3 {
	float m[3][3];
public:
	mat3() {}
	mat3(float m00, float m01, float m02,
		float m10, float m11, float m12,
		float m20, float m21, float m22) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
	}
	mat3 operator*(const mat3& right) {
		mat3 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	vec3 operator*(const vec3& right){
		vec3 result;
		result.x = m[0][0] * right.x + m[1][0] * right.y + m[2][0] * right.z;
		result.y = m[0][1] * right.x + m[1][1] * right.y + m[2][1] * right.z;
		result.z = m[0][2] * right.x + m[1][2] * right.y + m[2][2] * right.z;
		return result;
	}
	operator float*() { return &m[0][0]; }
};
#pragma endregion
#pragma region SHADERS
void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";
unsigned int shaderProgram;
class FullScreenTexturedQuad {
	unsigned int vao, textureId;
public:
	void Create(vec3 image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		static float vertexCoords[] = { -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glGenTextures(1, &textureId);  			
		glBindTexture(GL_TEXTURE_2D, textureId); 

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}
};
FullScreenTexturedQuad fullScreenTexturedQuad;
#pragma endregion
#pragma region RAY-THINGS
//RAY////////////////////////////////////////////////////////////
struct Ray {
public:
	vec3 start,	dir;
	Ray(vec3 point, vec3 direction) {
		start = point;
		dir = direction;
	}
};
//CAMERA/////////////////////////////////////////////////////////
class Camera {
public:
	vec3 pos;
	Camera(vec3 position) {
		this->pos = position;
	}
	Camera() {}
};
//LIGHT//////////////////////////////////////////////////////////
class Light {
public:
	vec3 color;
	vec3 pos;
	Light(vec3 position=vec3(0.0,0.0,0.0), vec3 color = vec3(0.0, 0.0, 0.0)) {
		this->pos = position;
		this->color = color;
	}
};
#pragma endregion
#pragma region OBJECTS
//MATERIAL///////////////////////////////////////////////////////
class Material {
public:
	bool istextured,  isRough,  isTukrozo, isFenytoro;
	vec3 Color;
	virtual vec3 reflect(vec3 inDir, vec3 normal)=0;
	virtual vec3 refract(vec3 inDir, vec3 normal)=0;
	virtual vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad)=0;
	virtual vec3 Fresnel(vec3 inDir, vec3 normal) = 0;

	Material(vec3 color = vec3(0, 0, 0), bool isTextured = false,bool isRough=true, bool isTukrozo=false,bool isFenytoro=false) {
		this->isRough= isRough;
		this->isTukrozo= isTukrozo;
		this->isFenytoro= isFenytoro;
		this->Color = color;
		this->istextured = isTextured;
	}
};
class MaterialRough :public Material {
public:
	vec3 kd, ks;
	float shininess;
	MaterialRough(vec3 color, bool textured, vec3 ks, vec3 kd, float shininess) :Material(color, textured, true, false, false){
		this->kd = kd;
		this->ks = ks;
		this->shininess = shininess;
	}
	vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir,vec3 inRad)	{
		vec3 reflRad(0, 0, 0);
		float cosTheta = dot(normal, lightDir);
		if (cosTheta < 0) return reflRad;
		reflRad = inRad * kd * cosTheta;
		vec3 halfway = (viewDir + lightDir).normalize();
		float cosDelta = dot(normal, halfway);
		if (cosDelta < 0) return reflRad;
		return reflRad + inRad * ks * pow(cosDelta, shininess);
	}
	virtual vec3 reflect(vec3 inDir, vec3 normal) { return vec3(0, 0, 0); }
	virtual vec3 refract(vec3 inDir, vec3 normal) { return vec3(0, 0, 0); }
	virtual vec3 Fresnel(vec3 inDir, vec3 normal) { return vec3(0, 0, 0); }

};
class MaterialSmooth:public Material {
public:
	vec3   f, n, k;
	float navg;
	MaterialSmooth(vec3 color, bool textured, vec3 n, vec3 k, bool isTukrozo, bool isFenytoro) :Material(color, textured, false, isTukrozo, isFenytoro) {
		this->n = n;
		this->k = k;
		this->navg = (n.x + n.y + n.z) / 3.0;
		f = ((n - vec3(1, 1, 1))*(n - vec3(1, 1, 1)) + k*k) / ((n + vec3(1, 1, 1))*(n + vec3(1, 1, 1)) + k*k);
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	}
	vec3 refract(vec3 inDir, vec3 normal) {
		float ior = navg;
		float cosa = -dot(normal, inDir);
		if (cosa < 0) { cosa = -cosa; normal = -normal; ior = 1 / navg; }
		float disc = 1 - (1 - cosa * cosa) / ior / ior;
		if (disc < 0) return reflect(inDir, normal);
		return inDir / ior + normal * (cosa / ior - sqrt(disc));
	}
	vec3 Fresnel(vec3 inDir, vec3 normal) {
		//TODO:Fresnel
		float cosa = fabs(dot(normal, inDir));
		return f + (vec3(1.0, 1.0, 1.0) - f)*pow(1 - cosa, 5);
	}
	virtual vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad) { return vec3(0, 0, 0); }
};
//HIT///////////////////////////////////////////////////////////
struct Hit {
	vec3 position, normal;
	Material* material;
	Object* object;
	float t;
	Hit() { t = -1; }
};
//OBJECT/////////////////////////////////////////////////////////
class Object {
public:
	Material* mat;

	virtual Hit intersect(Ray r) = 0;
	virtual vec3 getTextureInPoint(vec3 pos) = 0;
	Object(Material* mat) {
		this->mat = mat;
	}
};
///////////////////////////////////////////////////////HAMORSZOG//
class Triangle : public Object {
public:
	vec3 a, b, c, n;
	Triangle( vec3 a, vec3 b, vec3 c, Material* mat):Object(mat){
		this->a=a;
		this->b=b;
		this->c=c;
		n = cross(b - a, c - a).normalize();
	}
	Hit intersect(Ray ray)
	{
		Hit hit;
		float cost = dot(ray.dir, n);
		if (fabs(cost) < Epsilon) return hit;
		float t = (float)(dot((a - ray.start), n) / cost);
		if (t < Epsilon) return hit;
		vec3 p = ray.start + ray.dir * t;
		float c1 = dot(cross(b - a, p - a), n);
		float c2 = dot(cross(c - b, p - b), n);
		float c3 = dot(cross(a - c, p - c), n);
		if (c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0) {
			hit.position = p;
			hit.t = t; 
			hit.normal = n;
			hit.object = this;
			hit.material = mat;
		}
		return hit;
	}
	void move(vec3 newCenter) {
		a = a + newCenter;
		b = b + newCenter;
		c = c + newCenter;
	}
	void rotate(float A, float B, float C) {
		mat3 rmx( (float)( cosf(B)*cosf(C) ),							 (float)( -cosf(B)*sinf(C) ),						   (float) sinf(B),
				  (float)( sinf(A)*sinf(B) + cosf(A)*sinf(C) ),			 (float)( -sinf(A)*sinf(B) + cosf(A)*cosf(C) ),		   (float)( -sinf(A)*cosf(B) ),
				  (float)( -cosf(A)*sinf(B)*cosf(C) + sinf(A)*sinf(C) ), (float)( cosf(A)*sinf(B)*sinf(C) + sinf(A)*cosf(C) ), (float)( cosf(A)*cosf(B) )  );
		a = rmx*a;
		b = rmx*b;
		c = rmx*c;
		n = cross(b - a, c - a).normalize();
	}
	vec3 getTextureInPoint(vec3 pos) { return vec3(0, 0, 0); };
};
//////////////////////////////////////////////////////////TORUSZ//
class Torus{
public:
	Material* mat;
	float R,r,Rstep, rstep;
	int Rpart, rpart;
	Torus(float R, float r, int Rpart, int rpart, Material* mat){
		this->r = r;
		this->R = R;
		this->Rpart = Rpart;
		this->rpart = rpart;
		this->mat = mat;
		int hdb = 0;
		Rstep = (2 * M_PI) / Rpart;
		rstep = (2 * M_PI) / rpart;
	
	}
	vec3 TorusTo3D(vec3 p) {
		return vec3((R + r * cos(p.x)) *cos(p.y), (R + r * cos(p.x)) *sin(p.y), r* sin(p.x));
	}
	int AddObjects(int objdb, vec3 center,vec3 rotate) {
		for (int i = 0; i < Rpart; i++)
		{
			for (int j = 0; j < rpart; j++)
			{
				Triangle* t = new Triangle(TorusTo3D(vec3( i     * Rstep, j* rstep, 0)), TorusTo3D(vec3( i     *Rstep, (j+1)*rstep, 0)), TorusTo3D(vec3((i + 1)*Rstep, j    *rstep,  0)),mat);
				Triangle* t2 = new Triangle(TorusTo3D(vec3((i + 1)*Rstep, j*rstep, 0)), TorusTo3D(vec3((i + 1)*Rstep, (j+1)*rstep, 0)) , TorusTo3D(vec3( i     *Rstep, (j+1)*rstep,  0)),mat);
				t->rotate(rotate.x, rotate.y, rotate.z);
				t->move(center);
				t2->rotate(rotate.x, rotate.y, rotate.z);
				t2->move(center);
				objects[objdb++] = t;
				objects[objdb++] = t2;
			}
		}
		return (objdb-1);
	}
	vec3 getTextureInPoint(vec3 pos) { return vec3(0, 0, 0); };
};
////////////////////////////////////////////////////////////ROOM//
class Room : public Object {
public:
	vec3 center;
	float R;
	Room(vec3 center, double r, Material* mat):Object(mat) {
		this->center = center;
		this->R = r;
	}
	Hit intersect(Ray ray){
		Hit hit;
		float a = pow(ray.dir.x, 2.0)+pow(ray.dir.z,2.0);
		float b = 2 * (((ray.start.x - center.x) * ray.dir.x) + ((ray.start.z - center.z) * ray.dir.z));
		float c = pow((ray.start.x - center.x),2.0) + pow((ray.start.z - center.z) ,2.0) - pow(R,2.0);
		float discriminant = (b * b) - (4 * a * c);
		if (discriminant < 0.0) {
			return hit;
		}
		float t = (((-1.0 * b) + sqrt(discriminant)) / (2.0 * a));
		if (t > Epsilon) {
			hit.t = t;
			hit.material = mat;
			hit.position= ray.start + ray.dir * t;
			hit.normal = (center-hit.position).normalize();
			hit.normal.y = 0.0;
			hit.object = this;
			return hit;
		}
		return hit;
	}
	vec3 getTextureInPoint(vec3 pos) { 
		vec3 inUV = XYZtoUV(pos);
		inUV.x *= 600;
		if (fmodf(inUV.x, 50.0) <= 25) return vec3(0.353, 0.741, 0.866);
		else return vec3(0.64, 0.64, 0.64);
	}
	vec3 XYZtoUV(vec3 p){
		return vec3(acosf(p.x / R) / (2 * M_PI), p.y, 0.0);
	}
};
/////////////////////////////////////////////////PADLO ES PLAFON//
class RoomTopBottom : public Object {
public:
	vec3 point;
	vec3 normal;
	RoomTopBottom(vec3 p, vec3 norm, Material* mat): Object(mat) {
		point = p;
		normal = norm;
	}
	Hit intersect(Ray r){
		Hit hit;
		float cost = dot(r.dir, normal);
		if (fabs(cost) < Epsilon) return hit;
		float t = (float)(dot((point - r.start), normal) / cost);
		if (t < Epsilon) return hit;
		vec3 p = r.start + r.dir * t;
			hit.position = p;
			hit.t = t;
			hit.normal = normal;
			hit.object = this;
			hit.material = mat;
			return hit;
		return Hit();
	}
	vec3 getTextureInPoint(vec3 pos) {
		float a = (pos - point).Length();
		if (a < 50 || ((a > 100) && (a < 150)) || ((a > 200) && (a < 250)) || ((a > 300) && (a < 350)))  return vec3(0.353, 0.741, 0.866);
		else return vec3(0.75, 0.75, 0.75);
	}
};
////////////////////////////////////////////////////////////GOMB//
class Sphere : public Object {
public:
	vec3 origo;     //kozeppont
	double radius;  //sugar

	Sphere(vec3 o, double r, Material* mat) : Object(mat) {
		origo = o;
		radius = r;
	}

	Hit intersect(Ray ray) {
		double dx = ray.dir.x;
		double dy = ray.dir.y;
		double dz = ray.dir.z;
		//vec3 d = ray.dir;

		double x0 = ray.start.x;
		double y0 = ray.start.y;
		double z0 = ray.start.z;
		vec3 start = ray.start;

		double cx = origo.x;
		double cy = origo.y;
		double cz = origo.z;
		vec3 center = origo;

		double R = radius;

		double a = dx * dx + dy * dy + dz * dz;
		double b = 2 * dx * (x0 - cx) + 2 * dy * (y0 - cy) + 2 * dz * (z0 - cz);
		double c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0 - 2 * (cx * x0 + cy * y0 + cz * z0) - R * R;

		double d = b * b - 4 * a * c;
		Hit hit;
		/*if (d < 0) {
			return hit;
		}*/

		double t = ((-1.0 * b - sqrt(d)) / (2.0 * a));
		
		if (t > Epsilon) {
			vec3 p = ray.start + ray.dir * t;
			hit.position = p;
			hit.t = t;
			hit.normal = (p-origo).normalize(); //!!!
			hit.object = this;
			hit.material = mat;
			//printf("%f \t %f \t %f \n", p.x,p.y,p.z);
			return hit;
		}
		else {
			return hit;
		}
	}

	vec3 getNorm(vec3& intersect) {
		return (intersect - origo).normalize();
	}
	vec3 getTextureInPoint(vec3 pos) { return vec3(0, 0, 0); };
};

//SCENE///////////////////////////////////////////////////////////
class Scene{
public:
	Camera camera;
	Light* lights[3];
	int ldb=0;
	Scene() {}
	void Build() {
		camera.pos=vec3(0.0, 0.0, -500.0);
		Light* l1=  new Light(vec3(0.0,    250.0, 200.0), vec3(0.0, 0.0, 0.8) );
		Light* l2 = new Light(vec3(-180.0, 250.0, 50.0),  vec3(0.0, 0.8, 0.0) );
		Light* l3 = new Light(vec3(0.0,    290.0, 100.0), vec3(0.8, 0.0, 0.0) );
		lights[ldb++] = l1;
		lights[ldb++] = l2;
		lights[ldb]   = l3;
		
		Material* glass  = new MaterialSmooth(vec3(0.0, 0.0, 0.0),  false, vec3(1.5, 1.5, 1.5),    vec3(0.0, 0.0, 0.0), false, true );
		Material* gold   = new MaterialSmooth(vec3(0.1, 0.9, 0.3),  false, vec3(0.17, 0.35, 1.5),  vec3(3.1, 2.7, 1.9), true,  false);
		Material* silver = new MaterialSmooth(vec3(0.1, 0.1, 0.1),  false, vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1), true,  false);
		Material* wall   = new MaterialRough (vec3(1.0, 0.8, 0.43), true,  vec3(1, 1, 1),          vec3(0.5, 0.5, 0.5), 10);
		Material* roof   = new MaterialRough (vec3(1.0, 1.0, 1.0),  true,  vec3(0.5, 0.5, 0.5),    vec3(0.5, 0.5, 0.5), 10);
		Material* floor  = new MaterialRough (vec3(0.1, 0.24, 0.16),true,  vec3(0.5, 0.5, 0.5),    vec3(0.5, 0.5, 0.5), 10);

		Room* room            = new Room( vec3(0.0, 0.0, 100.0), 340.0, wall);
		RoomTopBottom* top    = new RoomTopBottom( vec3(0.0, 300.0, 100.0), vec3(0.0, -1.0, 0.0), roof);
		RoomTopBottom* bottom = new RoomTopBottom( vec3(0.0, -300.0, 0.0),  vec3(0.0, 1.0, 0.0),  floor);
		Torus* torus  = new Torus(150.0, 30.0, 30, 20, glass);
		Torus* torus2 = new Torus(100.0, 30.0, 30, 20, gold);
		Torus* torus3 = new Torus(100.0, 30.0, 30, 20, silver);
		Sphere* gomb1 = new Sphere(vec3(250,200,100),100, gold);
		Sphere* gomb2 = new Sphere(vec3(-250, -200, 100), 100, silver);
		Sphere* gomb3 = new Sphere(vec3(-60, 150, 100), 70, glass);
		Sphere* gomb4 = new Sphere(vec3(90, 150, 100), 55, glass);
		Sphere* gomb5 = new Sphere(vec3(-70, -200, 100), 50, glass);
		Sphere* gomb6 = new Sphere(vec3(60, -200, 100), 60, glass);


		objects[objdb++] = top;
		objects[objdb++] = bottom;
		objects[objdb++] = room;
		objects[objdb++] = gomb1;
		objects[objdb++] = gomb2;
		objects[objdb++] = gomb3;
		objects[objdb++] = gomb4;
		objects[objdb++] = gomb5;
		objects[objdb++] = gomb6;

		objdb = torus ->AddObjects(objdb, vec3(0.0,   -100.0, 300.0),   vec3(M_PI/8,  M_PI/4, M_PI / 2 ) );
		objdb = torus2->AddObjects(objdb, vec3(120.0, -180.0, 270.0), vec3(-2*M_PI/5, M_PI/8, M_PI/10  ) );
		objdb = torus3->AddObjects(objdb, vec3(-140.0, -50.0, 300.0), vec3(-M_PI/4,   M_PI/3, -M_PI/10 ) );
	}
	void Render(vec3* array) {
		for (int x = 0; x < windowWidth; x++) {
			for (int y = 0; y < windowHeight; y++) {
				vec3 p(x-300.5,y-300.5,0.0);
				//vec3 p(x - 960.5, y - 540.5, 0.0);
				Ray r = Ray(p, (p - camera.pos).normalize());
				array[y * windowWidth + x] = Trace(r,0);
				//array[y * windowWidth + x] = vec3((float)x / windowWidth, (float)y / windowHeight, 0);
			}
		}
	}
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (int i = 0; i <= objdb; i++)
		{
			Hit hit = objects[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
				bestHit = hit;
			}
		}
		return bestHit;
	}
	vec3 Trace(Ray& r,int depth){
		vec3 La = vec3(0.7, 0.7, 0.7);
		vec3 outRadiance(0.001,0.001, 0.001);

		if (depth++ > 10)
			return La;
		Hit hit = firstIntersect(r);
		if (hit.t < 0.0) return La;
		
		if (hit.material->isRough) {
			if (hit.material->istextured)
				outRadiance = hit.object->getTextureInPoint(hit.position)*La;
			else outRadiance = hit.material->Color * La;
			for (int i=0; i <= ldb; i++){
				Ray shadowRay(hit.position/* + (hit.normal*Epsilon)*/, (lights[i]->pos - hit.position).normalize());
				Hit shadowHit = firstIntersect(shadowRay);
				if (shadowHit.t < 0.0 || shadowHit.t >  (hit.position - lights[i]->pos).Length());
					outRadiance= outRadiance + hit.material->shade(hit.normal,-r.dir, (lights[i]->pos-hit.position).normalize() , lights[i]->color);
			}
		}
		if (hit.material->isTukrozo) {
			vec3 reflectionDir = hit.material->reflect(r.dir, hit.normal);
			Ray reflectedRay(hit.position /*+ (hit.normal*Epsilon)*/, reflectionDir);
			outRadiance =outRadiance+Trace(reflectedRay, depth++)*hit.material->Fresnel(-r.dir, hit.normal);
		}
		if (hit.material->isFenytoro) {
			vec3 refractionDir = hit.material->refract(r.dir, hit.normal);
			Ray refractedRay(hit.position/*+(hit.normal*Epsilon)*/, refractionDir);
			outRadiance = outRadiance+Trace(refractedRay, depth ++)*(vec3(1, 1, 1) - hit.material->Fresnel(-r.dir,hit.normal));
		}
		return outRadiance;
	}
};
#pragma endregion
#pragma region LISTENER
vec3 background[windowWidth * windowHeight];
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	Scene scene;
	scene.Build();
	scene.Render(background);
	fullScreenTexturedQuad.Create(background);
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");

	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	glUseProgram(shaderProgram);
}
void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay(); 
}
void onKeyboardUp(unsigned char key, int pX, int pY) {

}
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) { 
	}
}
void onMouseMotion(int pX, int pY) {
}
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
}
#pragma endregion
int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(100, 100);							
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE); 
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);           
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

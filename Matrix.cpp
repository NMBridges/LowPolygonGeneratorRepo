#include <iostream>
#include "Vector3.cpp"
#include "Vector2.cpp"
#include "Vector4.cpp"

class Matrix
{
private:



public:
	Vector3 a;
	Vector3 b;
	Vector3 c;

	Matrix()
	{
		a = Vector3();
		b = Vector3();
		c = Vector3();
	}

	Matrix(Vector3 one, Vector3 two, Vector3 three)
	{
		a = one;
		b = two;
		c = three;
	}

	double determinant()
	{
		double i = (b.y * c.z - b.z * c.y) * a.x;
		double j = (b.x * c.z - b.z * c.x) * a.y;
		double k = (b.x * c.y - b.y * c.x) * a.z;
		return (i - j + k);
	}


};
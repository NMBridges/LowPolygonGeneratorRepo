#include <iostream>
#include <math.h>
#include <string>

class Vector3
{
public:
	double x;
	double y;
	double z;

	Vector3()
	{
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}

	Vector3(double inx, double iny, double inz)
	{
		x = inx;
		y = iny;
		z = inz;
	}

	void subtract(Vector3 b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
	}

	void add(Vector3 b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
	}

	bool isZero()
	{
		return (x < 0.0001 && x > -0.0001 && y < 0.0001 && y > -0.0001 && z < 0.0001 && z > -0.0001);
	}

	std::string toString()
	{
		std::string var = "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
		return var;
	}
};
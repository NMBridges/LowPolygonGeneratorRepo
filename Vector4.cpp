#include <iostream>
#include <math.h>
#include <string>

class Vector4
{
public:
	double x;
	double y;
	double z;
	double w;

	Vector4()
	{
		x = 0.0;
		y = 0.0;
		z = 0.0;
		w = 0.0;
	}

	Vector4(double inx, double iny, double inz, double inw)
	{
		x = inx;
		y = iny;
		z = inz;
		w = inw;
	}

	void subtract(Vector4 b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
		w -= b.w;
	}

	void add(Vector4 b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
		w += b.w;
	}

	bool isZero()
	{
		return (x < 0.0001 && x > -0.0001 && y < 0.0001 && y > -0.0001 && z < 0.0001 && z > -0.0001 && w < 0.0001 && w > -0.0001);
	}

	std::string toString()
	{
		std::string var = "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(w) + ")";
		return var;
	}
};
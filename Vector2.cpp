#include <iostream>
#include <math.h>
#include <string>

class Vector2
{
public:
	double x;
	double y;

	Vector2()
	{
		x = 0.0;
		y = 0.0;
	}

	Vector2(double inx, double iny)
	{
		x = inx;
		y = iny;
	}

	void subtract(Vector2 b)
	{
		x -= b.x;
		y -= b.y;
	}

	void add(Vector2 b)
	{
		x += b.x;
		y += b.y;
	}

	bool isZero()
	{
		return (x < 0.0001 && x > -0.0001 && y < 0.0001 && y > -0.0001);
	}

	std::string toString()
	{
		std::string var = "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
		return var;
	}
};
#include <iostream>
#include <math.h>
#include <string>

class Vector3Int
{
public:
	int x;
	int y;
	int z;

	Vector3Int()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	Vector3Int(int inx, int iny, int inz)
	{
		x = inx;
		y = iny;
		z = inz;
	}

	void subtract(Vector3Int b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
	}

	void add(Vector3Int b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
	}

	bool isZero()
	{
		return (x == 0 && y == 0 && z == 0);
	}

	std::string toString()
	{
		std::string var = "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
		return var;
	}
};
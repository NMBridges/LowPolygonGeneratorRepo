#include <iostream>
#include <math.h>
#include <string>

class Vector3Int
{
public:
	uint16_t x;
	uint16_t y;
	uint16_t z;

	Vector3Int()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	Vector3Int(uint16_t inx, uint16_t iny, uint16_t inz)
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
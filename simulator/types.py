import numpy as np
import math

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point3D(self.x * other, self.y * other, self.z * other)
        raise NotImplementedError("Multiplication only supported with scalars")
    
    def __add__(self, other):
        if isinstance(other, Point3D):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        raise NotImplementedError("Addition only supported with another Point3D")
    
    def __sub__(self, other):
        if isinstance(other, Point3D):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        raise NotImplementedError("Subtraction only supported with another Point3D")
    
    def dot(self, other):
        if isinstance(other, Point3D):
            return self.x * other.x + self.y * other.y + self.z * other.z
        raise NotImplementedError("Dot product only supported with another Point3D")
    
    def distance_to(self, other):
        if isinstance(other, Point3D):
            dx = self.x - other.x
            dy = self.y - other.y
            dz = self.z - other.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            return Point3D(dx, dy, dz), dist
        raise NotImplementedError("Distance calculation only supported with another Point3D")
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
class State:
    def __init__(self, loc: Point3D, vel: Point3D, accel: Point3D, timestamp):
        self.loc = loc
        self.vel = vel
        self.accel = accel
        self.timestamp = timestamp

    def distance_to(self, other):
        if isinstance(other, State):
            return self.loc.distance_to(other.loc)
        raise NotImplementedError("Distance calculation only supported with another State")

    
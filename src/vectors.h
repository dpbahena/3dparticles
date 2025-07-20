#pragma once
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>




class Vec2 {
    public:
        float x, y;
        __host__ __device__
        Vec2() {}
        __host__ __device__
        Vec2(float v) : x(v), y(v) { };
        __host__ __device__
        Vec2 (float x, float y) : x(x), y(y){}
        
        __host__ __device__
        Vec2& operator=(const Vec2& v)  {
            x = v.x;
            y = v.y;
            return *this;
        }
        __host__ __device__
        bool operator==(const Vec2& v) const {
            return  (x == v.x && y == v.y);
        }

        // bool operator!=(const Vec2& v) const {
        //     return (x != v.x || y != v.y);
        // }
        __host__ __device__
        bool operator!=(const Vec2& v) const {
            return !(*this == v);  // Reuse operator==
        }
        __host__ __device__
        Vec2 operator+(const Vec2& other) const {
            return Vec2(other.x + x, other.y + y);
        }
        __host__ __device__
        Vec2 operator-(const Vec2& other) const {
            return Vec2(x - other.x, y - other.y);
        }
        __host__ __device__
        Vec2 operator*(const float n) const {
            return Vec2(x * n, y * n);
        }
        __host__ __device__
        Vec2 operator/(const float n) {
            return Vec2(x / n, y / n);
        }

        Vec2 operator-() const {   // negation
            return Vec2(-x, -y);
        };
        __host__ __device__
        Vec2& operator+=(const Vec2& other) {
            x += other.x;
            y += other.y;
            return *this;
        }
        __host__ __device__
        Vec2& operator-=(const Vec2& other) {
            x -= other.x;
            y -= other.y;
            return *this;
        }     
        __host__ __device__
        Vec2& operator*=(float n) {
            x*=n;
            y*=n;
            return *this;
        }
        __host__ __device__
        Vec2& operator/=(float n) {
            x/=n;
            y/=n;
            return *this;
        }
        
       
        __host__ __device__
        void scale(float s) {
            x *= s;
            y *= s;
            
        }
        
        // Vec2 rotate(const float a) const {
        //     float angle = toRadians(a);
        //     Vec2 vec{0,0};
        //     vec.x = x * cos(angle) - y * sin(angle);
        //     vec.y = x * sin(angle) + y * cos(angle);
        //     return vec;
        // }
        __device__ __host__
        float mag() const {
            return sqrt(x*x + y*y);
        }

        float magSquared() const {
            return (x*x + y*y);
        }
        __host__ __device__
        Vec2& normalize() {
            float length = this->mag();
            if (length != 0.0f) {
                x /= length;
                y /= length;
            }
            return *this;
        }

        Vec2 unitVector() const {
            float len = this->mag();
            if (len != 0.0f)
                return Vec2(x / len, y / len);
            else
                return Vec2(0.0,0.0);
        }

        Vec2 normal() const {
            return Vec2(y, -x).normalize();
        }
        __host__ __device__
        float dot(const Vec2& v) const {
            return x * v.x + y * v.y;
        }

        float cross(Vec2& v) const {
            return  (x * v.y - v.x * y);  // z-axis component only (determinant) - magnitud of the Z component
        }

        



        static Vec2 add(Vec2 v1, Vec2 v2) {
            return Vec2(v1.x + v2.x, v1.y + v2.y);
        }
        
        // void log(std::string name){
        //     printf("%s: {%f, %f}\n", name.c_str(), x, y);
        // }

        bool hasNaN() const {
            return std::isnan(x) || std::isnan(y);
        }
};

__host__ __device__
inline Vec2 operator*(const float n, const Vec2 &v) {
    return Vec2(v.x * n, v.y * n);
}



class Vec3 {
public:
    float x, y, z;

    // Constructors
    __host__ __device__
    Vec3() : x(0), y(0), z(0) {}
    __host__ __device__
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    // Magnitude
    __host__ __device__
    float mag() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    // Normalize
    __host__ __device__
    Vec3 normalize() const {
        float length = mag();
        if (length == 0.0f) return *this;
        return Vec3(x / length, y / length, z / length);
    }

    // Dot product
    __host__ __device__
    float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Cross product
    __host__ __device__
    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    // Scale
    __host__ __device__
    void scale(float s) {
        x *= s; y *= s; z *= s;
    }

    // Add/subtract (in-place)
    __host__ __device__
    void add(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
    }

    __host__ __device__
    void subtract(const Vec3& v) {
        x -= v.x; y -= v.y; z -= v.z;
    }

    // Static addition
    __host__ __device__
    static Vec3 add(const Vec3& v1, const Vec3& v2) {
        return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }

    // --------------------
    // Operator Overloads
    // --------------------
    __host__ __device__
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    __host__ __device__
    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    __host__ __device__
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    __host__ __device__
    Vec3& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    __host__ __device__
    bool operator==(const Vec3& v) const { return x == v.x && y == v.y && z == v.z; }
    __host__ __device__
    bool operator!=(const Vec3& v) const { return !(*this == v); }

    __host__ __device__
    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};




class Vec4 {
public:
    float x, y, z, w;

    // Constructors
    __host__ __device__
    Vec4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    // Basic arithmetic methods
    __host__ __device__
    void add(const Vec4& v) {
        x += v.x; y += v.y; z += v.z; w += v.w;
    }

    __host__ __device__
    void subtract(const Vec4& v) {
        x -= v.x; y -= v.y; z -= v.z; w -= v.w;
    }

    __host__ __device__
    void scale(float s) {
        x *= s; y *= s; z *= s; w *= s;
    }

    __host__ __device__
    float mag() const {
        return std::sqrt(x*x + y*y + z*z + w*w);
    }

    __host__ __device__
    Vec4 normalize() const {
        float length = mag();
        if (length == 0.0f) return *this; // Avoid divide by zero
        return Vec4(x/length, y/length, z/length, w/length);
    }

    __host__ __device__
    float dot(const Vec4& v) const {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    __host__ __device__
    Vec4 cross(const Vec4& v) const {
        return Vec4(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x,
            0.0f // Cross product of 3D part; w = 0 for direction
        );
    }

    __host__ __device__
    Vec4 homogenize() const {
        return (w != 0.0f) ? Vec4(x/w, y/w, z/w, 1.0f) : *this;
    }

    // Static utility
    __host__ __device__
    static Vec4 add(const Vec4& v1, const Vec4& v2) {
        return Vec4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
    }

    // --------------------
    // Operator Overloads
    // --------------------

    // Add two vectors
    __host__ __device__
    Vec4 operator+(const Vec4& v) const {
        return Vec4(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    // Subtract two vectors
    __host__ __device__
    Vec4 operator-(const Vec4& v) const {
        return Vec4(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    // Multiply vector by scalar
    __host__ __device__
    Vec4 operator*(float s) const {
        return Vec4(x * s, y * s, z * s, w * s);
    }

    // Divide vector by scalar
    __host__ __device__
    Vec4 operator/(float s) const {
        return Vec4(x / s, y / s, z / s, w / s);
    }

    // Unary minus
    __host__ __device__
    Vec4 operator-() const {
        return Vec4(-x, -y, -z, -w);
    }

    // Compound assignments
    __host__ __device__
    Vec4& operator+=(const Vec4& v) {
        x += v.x; y += v.y; z += v.z; w += v.w;
        return *this;
    }

    __host__ __device__
    Vec4& operator-=(const Vec4& v) {
        x -= v.x; y -= v.y; z -= v.z; w -= v.w;
        return *this;
    }

    __host__ __device__
    Vec4& operator*=(float s) {
        x *= s; y *= s; z *= s; w *= s;
        return *this;
    }

    __host__ __device__
    Vec4& operator/=(float s) {
        x /= s; y /= s; z /= s; w /= s;
        return *this;
    }

    // Equality check
    __host__ __device__
    bool operator==(const Vec4& v) const {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }

    __host__ __device__
    bool operator!=(const Vec4& v) const {
        return !(*this == v);
    }

    // Stream output
    __host__ __device__
    friend std::ostream& operator<<(std::ostream& os, const Vec4& v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
        return os;
    }
};

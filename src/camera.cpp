#include "camera.h"
#include <assert.h>

void Camera::setOrtographicProjection(float left, float right, float top, float bottom, float near, float far)
{
    
    projectionMatrix.m[0][0] = 2.f / (right - left);
    projectionMatrix.m[1][1] = 2.f / (bottom - top);
    projectionMatrix.m[2][2] = 1.f / (far - near);
    projectionMatrix.m[3][0] = -(right + left) / (right - left);
    projectionMatrix.m[3][1] = -(bottom + top) / (bottom - top);
    projectionMatrix.m[3][2] = - near / (far - near);
}

void Camera::setPerspectiveProjection(float fovy, float aspect, float near, float far)
{
    assert(fabs(aspect - std::numeric_limits<float>::epsilon()) > 0.0f);
    const float tanHalfFovy = tan(fovy / 2.f);

    projectionMatrix.m[0][0] = 1.f / (aspect * tanHalfFovy);
    projectionMatrix.m[1][1] = 1.f / (tanHalfFovy);
    projectionMatrix.m[2][2] = (far + near) / (far - near);
    projectionMatrix.m[2][3] = (2 * far * near) / (far - near);
    projectionMatrix.m[3][2] = 1.f;
}

void Camera::setViewDirection(Vec3 position, Vec3 direction, Vec3 up)
{
    const Vec3 w = direction.normalize();
    const Vec3 u = w.cross(up).normalize();
    const Vec3 v = w.cross(up);


    viewMatrix.m[0][0] = u.x;
    viewMatrix.m[1][0] = u.y;
    viewMatrix.m[2][0] = u.z;
    viewMatrix.m[0][1] = v.x;
    viewMatrix.m[1][1] = v.y;
    viewMatrix.m[2][1] = v.z;
    viewMatrix.m[0][2] = w.x;
    viewMatrix.m[1][2] = w.y;
    viewMatrix.m[2][2] = w.z;
    viewMatrix.m[3][0] = -u.dot(position);
    viewMatrix.m[3][1] = -v.dot(position);
    viewMatrix.m[3][2] = -w.dot(position);

}

void Camera::setViewTarget(Vec3 position, Vec3 target, Vec3 up)
{
    setViewDirection(position, target - position, up);
}

void Camera::setViewYXZ(Vec3 position, Vec3 rotation)
{
    const float c3 = cos(rotation.z);
    const float s3 = sin(rotation.z);
    const float c2 = cos(rotation.x);
    const float s2 = sin(rotation.x);
    const float c1 = cos(rotation.y);
    const float s1 = sin(rotation.y);
    const Vec3 u{(c1 * c3 + s1 * s2 * s3), (c2 * s3), (c1 * s2 * s3 - c3 * s1)};
    const Vec3 v{(c3 * s1 * s2 - c1 * s3), (c2 * c3), (c1 * c3 * s2 + s1 * s3)};
    const Vec3 w{(c2 * s1), (-s2), (c1 * c2)};
    viewMatrix.m[0][0] = u.x;
    viewMatrix.m[1][0] = u.y;
    viewMatrix.m[2][0] = u.z;
    viewMatrix.m[0][1] = v.x;
    viewMatrix.m[1][1] = v.y;
    viewMatrix.m[2][1] = v.z;
    viewMatrix.m[0][2] = w.x;
    viewMatrix.m[1][2] = w.y;
    viewMatrix.m[2][2] = w.z;
    viewMatrix.m[3][0] = -u.dot(position);
    viewMatrix.m[3][1] = -v.dot(position);
    viewMatrix.m[3][2] = -w.dot(position); 
}

void Camera::setLightDirection(Vec3 light)
{
    lightDirection = light;
}

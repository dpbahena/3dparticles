#pragma once
#include "vectors.h"




class MgeCamera {
public:

// main functions

    void setOrtographicProjection(float left, float right, float top, float bottom, float near, float far);
    void setPerspectiveProjection(float fovy, float aspect, float near, float far);               // fovy is the vertical field of view;

    void setViewDirection(Vec3 position, Vec3 direction, Vec3 up = Vec3{0.f, -1.f, 0.f} );
    void setViewTarget(Vec3 position, Vec3 target, Vec3 up = Vec3{0.f, -1.f, 0.f} );
    // set the orientation of the simulated camera
    void setViewYXZ(Vec3 position, Vec3 rotation); 
    void setLightDirection(Vec3 light);
    void setWireFrameMode(bool mode) { wireFrameMode = mode;}
            

// helpers functions
    
    

// getters

    const Mat4& getProjection() const { return projectionMatrix; }
    const Mat4& getView() const { return viewMatrix; }
    Vec3 getLightDirection() const {return lightDirection;}
    bool getWireFrameMode() const {return wireFrameMode; }

    
// setters

private:
// variable members

    Mat4 projectionMatrix; // identity by default
    Mat4 viewMatrix;       // identity by default
    Vec3 lightDirection;
    bool wireFrameMode = false;

    
// main functions
    

// helpers


};


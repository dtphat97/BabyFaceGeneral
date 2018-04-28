#pragma once
#include "stdafx.h"
#define DLIB_JPEG_SUPPORT
#include <opencv\cv.h>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <string>
#include <vector>

using namespace dlib;
using namespace std;
using namespace cv;

int getFeaturePoints(std::vector<Point2i> &PointList, string ImgName);
void getMorePoints(std::vector<Point2i> &PointList);
void getTriangulation(std::vector<Vec6f> &Triangle, std::vector<Point2i> &PointList, Size &ImgSize);
void morphBabyFromParents(Mat &Father, Mat &Mother, Mat &MorphImage, std::vector<Point2f> &TriangleFather, std::vector<Point2f> &TriangleMother, std::vector<Point2f> &TriangleMorph, double Alpha);

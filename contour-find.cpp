#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

/*
 * Find the angles between 2 line segments AB and BC, with common point B
 */
double angle(const Point a, const Point b, const Point c)
{
	/*
	 * Find the lengths AB and BC
	 * Use cosine law, desired angle = acos((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
	 */
	double AB = pow(pow(a.x - b.x, 2) + pow(a.y - b.y, 2), 0.5);
	double AC = pow(pow(a.x - c.x, 2) + pow(a.y - c.y, 2), 0.5);
	double BC = pow(pow(b.x - c.x, 2) + pow(b.y - c.y, 2), 0.5);

	double cos_angle = (AB * AB + BC * BC - AC * AC) / (2 * AB * BC);
	return acos(cos_angle);
}

bool isEllipse(const vector<Point> &contour, double threshold, bool &circle)
{
	/*
	 * Find the bounding rectangle
	 * Create a point set from the equation of the ellipse
	 * Translate all the points such that the center of the ellipse is at (0,0) and it is laying horizontal/virtical
	 * Compare that point set to the original contour
	 * If they are similar enough, return true
	 */
	RotatedRect bound = fitEllipse(contour);

	vector<Point> ellipse_points;
	Point2f center = bound.center;

	double a_2 = pow(bound.size.width * 0.5, 2);
	double b_2 = pow(bound.size.height * 0.5, 2);
	double ellipse_angle = (bound.angle * M_PI) / 180;

	size_t num_conforming_points = 0;
	for (const Point p: contour) {
		auto val = pow((p.x - center.x) * cos(ellipse_angle) + (p.y - center.y) * sin(ellipse_angle), 2) / a_2;
		val += pow((p.x - center.x) * sin(ellipse_angle) - (p.y - center.y) * cos(ellipse_angle), 2) / b_2;
		if (fabs(val - 1.0) < threshold)
			++num_conforming_points;
	}

	if (static_cast<double>(num_conforming_points) / contour.size() >= 0.5) {
		if (fabs(bound.size.width - bound.size.height) / 2 <= 2)
			circle = true;
		else
			circle = false;

		return true;
	}

	return false;
}

string getMemoryUsage()
{
	ifstream status_file("/proc/self/status");
	string line;

	while (getline(status_file, line)) {
		size_t pos;
		if ((pos = line.find("VmPeak:")) != string::npos)
			return line.substr(pos + 8, line.length());
	}

	return "";
}

bool findShapeFromContour(const vector<Point> &contour, string &shapeName, vector<Point> &detectedShape)
{
	vector<Point> approx;
	double max_angle = 0;
	approxPolyDP(contour, approx, /* epsilon */ arcLength(contour, true) * 0.02, /* curve closed */ true);

	// Shapes only appear with convex contours
	// Also, ignore shapes with area < <min_area>, because anything smaller is likely just noise / not an important find
	if (!isContourConvex(approx) || fabs(contourArea(approx)) <= 100)
		return false;

	switch (approx.size()) {
		case 0:
		case 1:
		case 2:
			break;
		case 3:
			// TODO: check if this is indeed a triangle
			shapeName = "triangle";
			detectedShape = approx;
			return true;
		case 4:
			for (size_t i = 0; i < 4; ++i) {
				// find the maximum cosine of the angle between joint edges
				double point_angle = fabs(angle(approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4]));
				max_angle = MAX(max_angle, point_angle);
			}
			if (fabs(max_angle - M_PI_2) < 0.1) {
				// All the angles are ~ pi/2, which means it is safe to assume it is a rectangle
				shapeName = "rectangle";
				detectedShape = approx;
				return true;
			}
			return false;
		case 5:
			//pentagon
			shapeName = "pentagon";
			detectedShape = approx;
			return true;
		default:
			// might still be an ellipse
			bool circle = false;
			if (isEllipse(contour, 0.09, circle)) {
				if (circle)
					shapeName = "circle";
				else
					shapeName = "ellipse";
				detectedShape = contour; //approximation looks bad for ellipses
				return true;
			}
			return false;
	}
	return false;
}

void findShapes(const Mat &image, vector<vector<Point>> &detectedShapes, vector<string> &shapeNames)
{
	/* Steps:
	 * 1. Pre-process
	 * 2. Find contours
	 * 3. Find shapes among contours
	 * 4. Report results
	 */

	Mat greyImage(image.size(), CV_8U), downscaledImage, upscaledImage;

	pyrDown(image, downscaledImage, Size(image.cols / 2, image.rows / 2));
	pyrUp(downscaledImage, upscaledImage, image.size());
	downscaledImage.release();

	// go one channel at a time (RGB)
	for (int c = 0; c < 3; ++c) {
		int ch[] = {c, 0};
		mixChannels(&upscaledImage, 1, &greyImage, 1, ch, 1);

		for (int level = 0; level < 11; ++level) {
			Mat thresholdedImage;
			if (level == 0) {
				// Find the edges, then dilate them to help the contour detector
				const int low_threshold = 10;
				const int high_threshold = 30;

				Canny(greyImage, thresholdedImage, low_threshold, high_threshold, 5);
				dilate(thresholdedImage, thresholdedImage, Mat(), Point(-1, -1));
			} else {
				thresholdedImage = greyImage >= (level + 1) * 255 / 10.0;
			}

			vector<vector<Point>> contours;

			findContours(thresholdedImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			for (auto& contour: contours) {
				string shapeName;
				vector<Point> shape;
				if (findShapeFromContour(contour, shapeName, shape)) {
					shapeNames.push_back(shapeName);
					detectedShapes.push_back(shape);
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		cerr << "usage: " << argv[0] << " <image file>\n";
		return 1;
	}

	Mat image = imread(argv[1]);
	if (!image.data) {
		cerr << "failed to open the image file\n";
		return 2;
	}

	vector<vector<Point>> shapes;
	vector<string> shape_names;

	auto start_time = chrono::high_resolution_clock::now();
	findShapes(image, shapes, shape_names);
	auto end_time = chrono::high_resolution_clock::now();

	polylines(image, shapes, true, Scalar(0, 255, 255), 3, LINE_AA);

	for (size_t i = 0; i < shape_names.size(); ++i) {
		Moments m = moments(shapes[i]);
		auto cX = m.m10 / m.m00;
		auto cY = m.m01 / m.m00;
		putText(image, shape_names[i], Point(cX, cY), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(128, 128, 128));
	}

	long long time_spent = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
	cout << "Took " << time_spent << "us\n";

	cout << "Peak memory usage: " << getMemoryUsage() << '\n';
	imshow("Contours", image);
	waitKey(0);

	return 0;
}

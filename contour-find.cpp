#include <cmath>
#include <chrono>
#include <iostream>
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

bool isEllipse(const vector<Point> &contour, double threshold)
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

	for (size_t i = 0; i < bound.size.width; i++) {
		double x = -bound.size.width * 0.5 + i;
		double y_left = sqrt((1 - (x * x / a_2)) * b_2);

		//rotate
		//[ cos(seta) sin(seta)]
		//[-sin(seta) cos(seta)]
		cv::Point2f rotate_point_left;
		rotate_point_left.x = cos(ellipse_angle) * x - sin(ellipse_angle) * y_left;
		rotate_point_left.y = sin(ellipse_angle) * x + cos(ellipse_angle) * y_left;

		//trans
		rotate_point_left += center;

		//store
		ellipse_points.push_back(Point(rotate_point_left));
	}

	for (size_t i = 0; i < bound.size.width; i++) {
		double x = bound.size.width * 0.5 - i;
		double y_right = -sqrt((1 - (x * x / a_2)) * b_2);

		//rotate
		//[ cos(seta) sin(seta)]
		//[-sin(seta) cos(seta)]
		cv::Point2f rotate_point_right;
		rotate_point_right.x = cos(ellipse_angle) * x - sin(ellipse_angle) * y_right;
		rotate_point_right.y = sin(ellipse_angle) * x + cos(ellipse_angle) * y_right;

		//trans
		rotate_point_right += center;

		//store
		ellipse_points.push_back(Point(rotate_point_right));
	}

	vector<vector<Point>> ellipses;
	ellipses.push_back(ellipse_points);

	double a0 = matchShapes(contour, ellipse_points, CV_CONTOURS_MATCH_I1, 0);
	cout << a0 << '\n';
	if (a0 > threshold)
		return true;

	return false;
}

void findShapes(const Mat &image, vector<vector<Point>> &detectedShapes, vector<string> &shape_names)
{
	/* Steps:
	 * 1. Change image to greyscale
	 * 2. Find contours
	 * 3. Find shapes among contours
	 * 4. Report results
	 */

	Mat greyImage(image.size(), CV_8U), downscaled_image, upscaled_image;

	// Downscale and then upscale image to clear up some noise
	pyrDown(image, downscaled_image, Size(image.cols / 2, image.rows / 2));
	pyrUp(downscaled_image, upscaled_image, image.size());

	// go one channel at a time (RGB)
	for (int c = 0; c < 3; ++c) {
		int ch[] = {c, 0};
		mixChannels(&image, 1, &greyImage, 1, ch, 1);

		Mat detectedEdges;
		const int low_threshold = 0;
		const int high_threshold = 100;

		// Find the edges, then dilate them to help the contour detector
		Canny(greyImage, detectedEdges, low_threshold, high_threshold, 5);
		dilate(detectedEdges, detectedEdges, Mat(), Point(-1, -1));

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		findContours(detectedEdges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		hierarchy.clear();

		vector<Point> approx;
		double max_angle = 0;

		for (auto& contour: contours) {
			approxPolyDP(contour, approx, arcLength(contour, true) * 0.01, true);

			// Shapes only appear with convex contours
			// Also, ignore shapes with area < <min_area>, because anything smaller is likely just noise / not an important find
			if (!isContourConvex(approx) || fabs(contourArea(approx)) <= 100)
				continue;

			switch (approx.size()) {
				case 0:
				case 1:
				case 2:
					break;
				case 3:
					// TODO: check if this is indeed a triangle
					shape_names.push_back("triangle");
					detectedShapes.push_back(approx);
					break;
				case 4:
					for (size_t i = 2; i < 5; ++i) {
						// find the maximum cosine of the angle between joint edges
						double point_angle = fabs(angle(approx[i % 4], approx[i - 2], approx[i - 1]));
						max_angle = MAX(max_angle, point_angle);
					}
					if (max_angle < M_PI_2) {
						// All the angles are ~ pi/2, which means it is safe to assume it is a rectangle
						shape_names.push_back("rectangle");
						detectedShapes.push_back(approx);
					}
					break;
				case 5:
					//pentagon
					shape_names.push_back("pentagon");
					detectedShapes.push_back(approx);
					break;
				default:
					// might still be an ellipse
					if (isEllipse(contour, 0.001)) {
						shape_names.push_back("ellipse");
						detectedShapes.push_back(contour);
					}
					break;
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

	imshow("Contours", image);
	waitKey(0);

	return 0;
}

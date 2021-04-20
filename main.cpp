#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Find rectangles from the lines found by a hough transform
// Returns: groups of lines that form rectangles
// TODO: don't mutate lines, make it const
vector<Vec2f> findRectangles(vector<Vec2f> &lines)
{
	/*
	 * Sort the lines by theta (the angle between the line and the horizontal axis)
	 * Lines that are parrallel will have the same theta
	 * Find and return pairs that are perpendicular to each other (sum of thetas = pi/2)
	 */
	sort(lines.begin(), lines.end(), [](Vec2f a, Vec2f b) {
			float theta_a = a[1], theta_b = b[1];
			if (theta_a - theta_b < 0.1)
				return 1;
			else if (theta_b - theta_a < 0.1)
				return -1;
			return 0;
		});

	// The last three lines can't form a rectangle, so skip checking them
	vector<Vec2f> results;
	for (size_t i = 0; i < lines.size() - 3; ++i) {
		if (abs(lines[i][1] - lines[i + 1][1]) < 0.1) {
			float perpendicular_theta = abs(M_PI_2 - lines[i][1]); // theta of lines that are perpendicular to the lines we're look at

			// Find a list of perpendicular lines from list of all lines
			auto perpendicular_line = find_if(lines.begin() + i + 1, lines.end(), [=](Vec2f a) { return abs(a[1] - perpendicular_theta) < 0.1; });
			// Check if the perpendicular line we found has another line parallel to it

			Vec2f line1 = *perpendicular_line++;
			Vec2f line2 = *perpendicular_line;

			if (abs(line1[1] - line2[1]) < 0.1) {
				//line1 and line2 are parallel, and lines[1] and lines[1+1] are perpendicular to them
				results.push_back(lines[i]);
				results.push_back(lines[i + 1]);
				results.push_back(line1);
				results.push_back(line2);
			}
		}
	}

	return results;
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		cerr << "usage: " << argv[0] << " <image file>\n";
		return 1;
	}

	Mat colorMat = imread(argv[1]);
	if (!colorMat.data) {
		cerr << "failed to open the image file\n";
		return 2;
	}

	/* Steps:
	 * 1. Change image to greyscale
	 * 2. Find edges
	 * 3. Find shapes
	 * 4. Report results
	 */

	Mat greyMat;
	cvtColor(colorMat, greyMat, CV_BGR2GRAY);

	Mat detectedEdges;
	blur(greyMat, detectedEdges, Size(3, 3));
	int low_threshold = 30;
	Canny(detectedEdges, detectedEdges, low_threshold, 90, 3);

	vector<Vec2f> lines;
	// Switch to HoughLines2 for only segments
	HoughLines(detectedEdges, lines, 1, CV_PI / 180, 150, 0, 0);

	vector<Vec2f> rectangles = findRectangles(lines);
	cout << "Found " << rectangles.size() / 4 << " rectangles\n";

	vector<Vec3f> circles;
	int min_radius = 1, max_radius = 100;
	double min_dist = greyMat.rows / 16;

	HoughCircles(greyMat, circles, HOUGH_GRADIENT, 1,
			min_dist, 100, 30, min_radius, max_radius);
	cout << "Found " << circles.size() << " circles\n";

	return 0;
}

opencv_dir=/usr/include/opencv4

CXXFLAGS = -Wall -Wextra -pedantic -std=c++14 -I$(opencv_dir) -Ofast
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
TARGETS = build/contour_shape_detect build/hough_shape_detect

all: $(TARGETS)

build/hough_shape_detect: hough-find.cpp
	-mkdir -p build
	$(CXX) $(CXXFLAGS) hough-find.cpp $(LIBS) -o build/hough_shape_detect

build/contour_shape_detect: contour-find.cpp
	-mkdir -p build
	$(CXX) $(CXXFLAGS) contour-find.cpp $(LIBS) -o build/contour_shape_detect

clean:
	-rm -rf build/

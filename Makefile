opencv_dir=/usr/include/opencv4

CXXFLAGS = -Wall -Wextra -pedantic -std=c++14 -I$(opencv_dir) -Ofast
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
TARGETS = build/shape_detect

all: $(TARGETS)

build/shape_detect: hough_find.cpp
	-mkdir -p build
	$(CXX) $(CXXFLAGS) hough_find.cpp $(LIBS) -o build/hough_shape_detect

clean:
	-rm -rf build/

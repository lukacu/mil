PROJECT (BHMC)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
GET_FILENAME_COMPONENT(PROJECT_ROOT . ABSOLUTE)

FIND_PACKAGE(OpenCV REQUIRED)

find_file(TRAX_HEADER NAMES trax.h)
get_filename_component(TRAX_DIR ${TRAX_HEADER} DIRECTORY)
link_directories(${TRAX_DIR})

LINK_LIBRARIES(${OpenCV_LIBS}  trax trax_opencv)
INCLUDE_DIRECTORIES(AFTER ${OpenCV_INCLUDE_DIR} ${TRAX_DIR} "./include")

ADD_EXECUTABLE(mil ./src/ImageFtr.cpp ./src/Public.cpp ./src/Sample.cpp ./src/OnlineBoost.cpp ./src/Matrix.cpp ./src/Tracker.cpp ./src/BoostTrackMain.cpp)



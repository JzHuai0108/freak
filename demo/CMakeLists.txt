if( BUILD_DEMO )
    add_executable(freak_demo freak_demo.cpp)
    target_link_libraries(freak_demo freak_p
            ${OpenCV_LIBS} opencv_nonfree)
endif( BUILD_DEMO )

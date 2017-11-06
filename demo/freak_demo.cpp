//  demo.cpp
//
//	Here is an example on how to use the descriptor presented in the following paper:
//	A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.
//  CVPR 2012 Open Source Award winner
//
//	Copyright (C) 2011-2012  Signal processing laboratory 2, EPFL,
//	Kirell Benzi (kirell.benzi@epfl.ch),
//	Raphael Ortiz (raphael.ortiz@a3.epfl.ch),
//	Alexandre Alahi (alexandre.alahi@epfl.ch)
//	and Pierre Vandergheynst (pierre.vandergheynst@epfl.ch)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#if CV_MAJOR_VERSION==2
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "src/hammingseg.h"
#include "src/freak.h"
#elif CV_MAJOR_VERSION==3
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "src3/freak.h"
#endif

void help( char** argv )
{
    std::cout << "\nUsage: " << argv[0] << " [path/to/image1] [path/to/image2] \n"
              << "This is an example on how to use the keypoint descriptor presented in the following paper: \n"
              << "A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. \n"
              << "In IEEE Conference on Computer Vision and Pattern Recognition, 2012. CVPR 2012 Open Source Award winner \n"
              << std::endl;
}

int main( int argc, char** argv ) {
    // check http://opencv.itseez.com/doc/tutorials/features2d/table_of_content_features2d/table_of_content_features2d.html
    // for OpenCV general detection/matching framework details

    if( argc != 3 ) {
        help(argv);
        return -1;
    }

    // Load images
    cv::Mat imgA = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    if( !imgA.data ) {
        std::cout<< " --(!) Error reading image " << argv[1] << std::endl;
        return -1;
    }

    cv::Mat imgB = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
    if( !imgA.data ) {
        std::cout << " --(!) Error reading image " << argv[2] << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypointsA, keypointsB;
    cv::Mat descriptorsA, descriptorsB;
    std::vector<cv::DMatch> matches;

    keypointsA.clear();
    keypointsB.clear();

    // DETECTION
#if CV_MAJOR_VERSION==2
    // Any openCV detector such as
//    cv::SurfFeatureDetector detector(2000,4);
    cv::BRISK detector;

    // DESCRIPTOR
    // Our proposed FREAK descriptor
    // (roation invariance, scale invariance, pattern radius corresponding to SMALLEST_KP_SIZE,
    // number of octaves, optional vector containing the selected pairs)
    // FREAK extractor(true, true, 22, 4, std::vector<int>());
    freak::FREAK extractor(false);

#elif CV_MAJOR_VERSION==3
    cv::Ptr<cv::Feature2D> detector;
    detector = cv::BRISK::create();
    cv::Ptr<cv::Feature2D>  extractor = freak::FREAK::create(false);
#endif

    // MATCHER
#if CV_MAJOR_VERSION==2
    // The standard Hamming distance can be used such as
    // BruteForceMatcher<Hamming> matcher;
    // or the proposed cascade of hamming distance using SSSE3
#if CV_SSSE3
    cv::BruteForceMatcher< freak::HammingSeg<30,4> > matcher;
#else
    cv::BruteForceMatcher<cv::Hamming> matcher;
#endif
#elif CV_MAJOR_VERSION==3
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
#endif

    // detect
    double t = (double)cv::getTickCount();
#if CV_MAJOR_VERSION==2
    detector.detect( imgA, keypointsA );
    detector.detect( imgB, keypointsB );
#elif CV_MAJOR_VERSION==3
    detector->detect( imgA, keypointsA );
    detector->detect( imgB, keypointsB );
#endif
    for(size_t jack = 0; jack < keypointsA.size(); ++ jack)
    {
        std::cout <<jack << "kp angle "<< keypointsA[jack].angle<<std::endl;
    }
    cv::Mat imgABrisk;
    cv::drawKeypoints(imgA, keypointsA, imgABrisk, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    cv::namedWindow("brisk keys", CV_WINDOW_KEEPRATIO);
    cv::imshow("brisk keys", imgABrisk);


    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    std::cout << "detection time [s]: " << t/1.0 << std::endl;

    // extract
    t = (double)cv::getTickCount();
#if CV_MAJOR_VERSION==2
    extractor.compute( imgA, keypointsA, descriptorsA );
    extractor.compute( imgB, keypointsB, descriptorsB );
#elif CV_MAJOR_VERSION==3
    extractor->compute( imgA, keypointsA, descriptorsA );
    extractor->compute( imgB, keypointsB, descriptorsB );
#endif
    std::cout <<"After descriptor extraction "<<std::endl;
    for(size_t jack = 0; jack < keypointsA.size(); ++ jack)
    {
        std::cout <<jack << "kp angle "<< keypointsA[jack].angle<<std::endl;
    }

    cv::Mat imgAFreak;
    cv::drawKeypoints(imgA, keypointsA, imgAFreak, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    cv::namedWindow("freak keys", CV_WINDOW_KEEPRATIO);
    cv::imshow("freak keys", imgAFreak);


    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    std::cout << "extraction time [s]: " << t << std::endl;

    // match
    t = (double)cv::getTickCount();
    matcher.match(descriptorsA, descriptorsB, matches);
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    std::cout << "matching time [s]: " << t << std::endl;

    // Draw matches
    cv::Mat imgMatch;
    cv::drawMatches(imgA, keypointsA, imgB, keypointsB, matches, imgMatch,
      cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char> (), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


    cv::namedWindow("matches", CV_WINDOW_KEEPRATIO);
    cv::imshow("matches", imgMatch);
    cv::waitKey(0);
}

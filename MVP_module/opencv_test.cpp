
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace cv;
using namespace std;

// Vec3f mean = Vec3f(0.485, 0.456, 0.406)
// Vec3f std = Vec3f(0.229, 0.224, 0.225)
void image_normalize(const Mat& image, Mat& outImage, Vec3f mean, Vec3f std)
 {
     // 创建与原图像等尺寸的图像
    outImage.create(image.size(), CV_32FC3);    // image.type()

    for(int i=0;i<image.rows;i++)
    {
      for(int j=0;j<image.cols;j++)
      {
          outImage.at<Vec3f>(i,j) = Vec3f( ((image.at<Vec3b>(i,j)[0]/255.0)-mean[0])/std[0],\
                                           ((image.at<Vec3b>(i,j)[1]/255.0)-mean[1])/std[1],\
                                           ((image.at<Vec3b>(i,j)[2]/255.0)-mean[2])/std[2] );
          // image.at<Vec3b>(i,j)[0]/div*div+div/2;
          // image.at<Vec3b>(i,j)[1] = image.at<Vec3b>(i,j)[1]/div*div+div/2;
          // image.at<Vec3b>(i,j)[2] = image.at<Vec3b>(i,j)[2]/div*div+div/2;
      }
    }
}

//功能：对图片进行特效处理并显示
int main(int argc,char **argv)
{
    cv::Mat img = imread("C0.jpg");

    auto start_read = std::chrono::system_clock::now();
    for(int i=0; i<50; i++) {

        cv::Mat img_float;
        img.convertTo(img_float, CV_32FC3);
        img_float = img_float.mul(cv::Scalar(1 / 255.0, 1 / 255.0, 1 / 255.0));
        img_float = img_float - cv::Scalar(0.485, 0.456, 0.406);
        img_float = img_float.mul(cv::Scalar(1 / 0.229, 1 / 0.224, 1 / 0.225));
    }
    auto end_read = std::chrono::system_clock::now();
    std::cout << "normalize time1:" << std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count() << "ms" << std::endl;

    start_read = std::chrono::system_clock::now();
    for(int i=0; i<50; i++) {

        cv::Mat outImage;
        Vec3f mean = Vec3f(0.485, 0.456, 0.406);
        Vec3f std = Vec3f(0.229, 0.224, 0.225);
        image_normalize(img, outImage, mean, std);
    }
    end_read = std::chrono::system_clock::now();
    std::cout << "normalize time2:" << std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count() << "ms" << std::endl;

    return 0;
}




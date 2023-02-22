#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>

#include <iostream>
#include <math.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <detection_msgs/BoundingBox.h>
#include <detection_msgs/BoundingBoxes.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>

#include <chrono> 

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

//Publisher
ros::Publisher pc_filtered_pub; // publisher de la imagen de puntos filtrada

float maxlen =100.0;    //maxima distancia del lidar
float minlen = 0.01;    //minima distancia del lidar
float max_FOV = 3.0;    // en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;    // en radianes angulo minimo de vista de la camara

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x =0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width  = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float interpol_value = 20.0;

// input topics 
std::string reflecTopic = "/ouster/reflec_image";
std::string nearirTopic = "/ouster/nearir_image";
std::string signalTopic = "/ouster/signal_image";
std::string rangeTopic  = "/ouster/range_image";
std::string pcTopic     = "/ouster/points";
std::string objYoloTopic= "/yolov5/detections";

///////////////////////////////////////callback


void callback(const ImageConstPtr& in_image, 
              const boost::shared_ptr<const detection_msgs::BoundingBoxes> &bb_data)
{
    cv_bridge::CvImagePtr cv_range;
        try
        {
          cv_range = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::MONO16);
        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
        }

  cv::Mat img_range  = cv_range->image; // get image matrix of cv_range
  Eigen::MatrixXf data_aux(4,4);
  Eigen::Matrix<float,Dynamic,Dynamic> depth_data;
  Eigen::Matrix<float,Dynamic,Dynamic> data_metrics;  // matrix with image values and matrix qith image values into real range data
  //Eigen::MatrixXf depth_data(128,2048), data_metrics(128,2048);
  cv2eigen(img_range,depth_data);       // convert img_range into eigen matrix

  double minVal; 
  double maxVal; 
  cv::Point minLoc; 
  cv::Point maxLoc;

  minMaxLoc( img_range, &minVal, &maxVal, &minLoc, &maxLoc );

  cout << "min val: " << minVal << endl;
  cout << "max val: " << maxVal << endl;

  auto max_range = depth_data.maxCoeff();
  auto min_range = depth_data.minCoeff();
  std::cout<<"max_range: "<<max_range<< std::endl;
  std::cout<<"min_range: "<<min_range<< std::endl;

  data_metrics = depth_data;// * (200.0/65536.0);
  
  detection_msgs::BoundingBoxes data = *bb_data;
  uint num_yolo_detection = data.bounding_boxes.size();

  for(uint i=0 ;i<num_yolo_detection; i++)
	{
    uint xmin = data.bounding_boxes[i].xmin;
    uint ymin = data.bounding_boxes[i].ymin;
    uint xmax = data.bounding_boxes[i].xmax;
    uint ymax = data.bounding_boxes[i].ymax;

    float depth_ave = 0;   //  average distance of object
    int cont_pix=0;        // number of pixels 

    float bb_per = 0.4;  // bounding box reduction percentage 
    bb_per = bb_per/2;

    uint start_x = (1-bb_per) * xmin + (bb_per * xmax);
    uint end_x   = (1-bb_per) * xmax + (bb_per * xmin);
    uint start_y = (1-bb_per) * ymin + (bb_per * ymax);
    uint end_y   = (1-bb_per) * ymax + (bb_per * ymin);
 
    for (uint iy = start_y;iy<end_y; iy++)
      for (uint ix = start_x;ix<end_x; ix++){

        if(data_metrics(iy,ix)>0){
          depth_ave += data_metrics(iy,ix);
          cont_pix++;
        }
      }
    
  depth_ave = depth_ave/cont_pix;
  std::cout<<"contador: "<<i<<"average: "<<depth_ave << std::endl;

  }





  /*

  P_out = cloud;

  int size_inter_Lidar = (int) P_out->points.size();

  Eigen::MatrixXf Lidar_camera(3,size_inter_Lidar);
  Eigen::MatrixXf Lidar_cam(3,1);
  Eigen::MatrixXf pc_matrix(4,1);
  Eigen::MatrixXf pointCloud_matrix(4,size_inter_Lidar);

  unsigned int cols = in_image->width;
  unsigned int rows = in_image->height;

  uint px_data = 0; uint py_data = 0;


  pcl::PointXYZRGB point;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color (new pcl::PointCloud<pcl::PointXYZRGB>);

   //P_out = cloud_out;

  for (int i = 0; i < size_inter_Lidar; i++)
  {
      pc_matrix(0,0) = -P_out->points[i].y;   
      pc_matrix(1,0) = -P_out->points[i].z;   
      pc_matrix(2,0) =  P_out->points[i].x;  
      pc_matrix(3,0) = 1.0;

      Lidar_cam = Mc * (RTlc * pc_matrix);

      px_data = (int)(Lidar_cam(0,0)/Lidar_cam(2,0));
      py_data = (int)(Lidar_cam(1,0)/Lidar_cam(2,0));
      
      if(px_data<0.0 || px_data>=cols || py_data<0.0 || py_data>=rows)
          continue;

      int color_dis_x = (int)(255*((P_out->points[i].x)/maxlen));
      int color_dis_z = (int)(255*((P_out->points[i].x)/10.0));
      if(color_dis_z>255)
          color_dis_z = 255;


      //point cloud con color
      cv::Vec3b & color = color_pcl->image.at<cv::Vec3b>(py_data,px_data);

      point.x = P_out->points[i].x;
      point.y = P_out->points[i].y;
      point.z = P_out->points[i].z;
      

      point.r = (int)color[2]; 
      point.g = (int)color[1]; 
      point.b = (int)color[0];

      
      pc_color->points.push_back(point);   
      
      cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, CV_RGB(255-color_dis_x,(int)(color_dis_z),color_dis_x),cv::FILLED);
      
    }
    pc_color->is_dense = true;
    pc_color->width = (int) pc_color->points.size();
    pc_color->height = 1;
    pc_color->header.frame_id = "velodyne";

  pc_filtered_pub.publish(cv_ptr->toImageMsg());
  pc_pub.publish (pc_color);*/

}

int main(int argc, char** argv)
{

  ros::init(argc, argv, "pontCloudOntImage");
  ros::NodeHandle nh;  
  std::cout<<"Nodo inicializado: "<<std::endl;
  
  /// Load Parameters

  nh.getParam("/maxlen", maxlen);
  nh.getParam("/minlen", minlen);
  nh.getParam("/pcTopic", pcTopic);
  nh.getParam("/reflec_img", reflecTopic);
  nh.getParam("/signal_img", signalTopic);
  nh.getParam("/nearir_img", nearirTopic);
  nh.getParam("/range_img", rangeTopic);
  nh.getParam("/x_resolution", angular_resolution_x);
  nh.getParam("/ang_Y_resolution", angular_resolution_y);
  nh.getParam("/detection_BoundingBoxes", objYoloTopic);
   
  // ==================== Synchronizer messages ======================
  //message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic , 1);
  /*message_filters::Subscriber<Image>reflec_sub(nh, reflecTopic, 1);
  message_filters::Subscriber<Image>signal_sub(nh, signalTopic, 1);
  message_filters::Subscriber<Image>nearir_sub(nh, nearirTopic, 1);*/
  message_filters::Subscriber<Image>range_sub (nh, rangeTopic,  10);
  message_filters::Subscriber<detection_msgs::BoundingBoxes> yoloBB_sub(nh, objYoloTopic , 10);

  typedef sync_policies::ApproximateTime<Image, detection_msgs::BoundingBoxes> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), range_sub, yoloBB_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  pc_filtered_pub = nh.advertise<PointCloud> ("/ouster_filtered", 1);  

  ros::spin();
}

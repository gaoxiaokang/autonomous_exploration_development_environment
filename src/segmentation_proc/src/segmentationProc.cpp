#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;

const double PI = 3.1415926;

string seg_file_dir;
int segmentationDisplayInterval = 2;
int segmentationDisplayCount = 0;

struct RegionPoint {
     float x, y, z;
     float box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;
     float height;
     int region_index;
     uint8_t label[1024];
};

POINT_CLOUD_REGISTER_POINT_STRUCT (RegionPoint,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, box_min_x, box_min_x)
                                   (float, box_min_y, box_min_y)
                                   (float, box_min_z, box_min_z)
                                   (float, box_max_x, box_max_x)
                                   (float, box_max_y, box_max_y)
                                   (float, box_max_z, box_max_z)
                                   (float, height, height)
                                   (int, region_index, region_index)
                                   (uint8_t[1024], label, label))

struct ObjectPoint {
     float x, y, z;
     float axis0_x, axis0_y, axis0_z;
     float axis1_x, axis1_y, axis1_z;
     float radius_x, radius_y, radius_z;
     int object_index, region_index, category_index;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (ObjectPoint,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, axis0_x, axis0_x)
                                   (float, axis0_y, axis0_y)
                                   (float, axis0_z, axis0_z)
                                   (float, axis1_x, axis1_x)
                                   (float, axis1_y, axis1_y)
                                   (float, axis1_z, axis1_z)
                                   (float, radius_x, radius_x)
                                   (float, radius_y, radius_y)
                                   (float, radius_z, radius_z)
                                   (int, object_index, object_index)
                                   (int, region_index, region_index)
                                   (int, category_index, category_index))

pcl::PointCloud<RegionPoint>::Ptr regionAll(new pcl::PointCloud<RegionPoint>());
pcl::PointCloud<ObjectPoint>::Ptr objectAll(new pcl::PointCloud<ObjectPoint>());

float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;

sensor_msgs::PointCloud2 regionAll2, objectAll2;

void odometryHandler(const nav_msgs::Odometry::ConstPtr& odom)
{
  double roll, pitch, yaw;
  geometry_msgs::Quaternion geoQuat = odom->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  vehicleX = odom->pose.pose.position.x;
  vehicleY = odom->pose.pose.position.y;
  vehicleZ = odom->pose.pose.position.z;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "segmentationProc");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");

  nhPrivate.getParam("seg_file_dir", seg_file_dir);
  nhPrivate.getParam("segmentationDisplayInterval", segmentationDisplayInterval);

  ros::Subscriber subOdometry = nh.subscribe<nav_msgs::Odometry> ("/state_estimation", 5, odometryHandler);

  ros::Publisher pubRegion = nh.advertise<sensor_msgs::PointCloud2> ("/region_segmentations", 5);

  ros::Publisher pubObject = nh.advertise<sensor_msgs::PointCloud2> ("/object_segmentations", 5);

  //////////////////////////////////////////////////////////////////////////////////////
  // read matterport.house file and parse segmentation data into regionAll and objectAll
  // publish both as visualization marker arrays to display in Rviz
  //////////////////////////////////////////////////////////////////////////////////////

  pcl::toROSMsg(*regionAll, regionAll2);
  pcl::toROSMsg(*objectAll, objectAll2);

  time_t systemTime = time(0);

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();

    segmentationDisplayCount++;
    if (segmentationDisplayCount >= 100 * segmentationDisplayInterval) {
      regionAll2.header.stamp = ros::Time().fromSec(systemTime);
      regionAll2.header.frame_id = "/map";
      pubRegion.publish(regionAll2);

      objectAll2.header.stamp = regionAll2.header.stamp;
      objectAll2.header.frame_id = "/map";
      pubObject.publish(objectAll2);

      segmentationDisplayCount = 0;
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}

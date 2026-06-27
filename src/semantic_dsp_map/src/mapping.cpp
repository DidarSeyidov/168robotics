/**
 * @file mapping.cpp
 * @author Clarence Chen (g-ch@github.com)
 * @brief An example of using the SemanticDSPMap in a ROS2 node
 * @version 0.2
 * @date 2023-12-12
 *
 * @copyright Copyright (c) 2023
 */

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <functional>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <settings/external_settings.h>
#include <cv_bridge/cv_bridge.h>

#include <mask_kpts_msgs/msg/mask_group.hpp>
#include <mask_kpts_msgs/msg/mask_kpts.hpp>
#include <mask_kpts_msgs/msg/keypoint.hpp>

#include <yaml-cpp/yaml.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "semantic_dsp_map.h"

class MappingNode : public rclcpp::Node
{
public:
    MappingNode(std::string yaml_file, std::string object_info_csv_file = "")
    : Node("mapping_with_external_data"),
      yaml_file_(yaml_file),
      object_info_csv_file_(object_info_csv_file)
    {
        initialize();
    }

    ~MappingNode() {}

private:
    cv::Mat depth_image_;
    Eigen::Vector3d current_camera_position_, last_camera_position_;
    Eigen::Quaterniond current_camera_orientation_, last_camera_orientation_;

    double time_stamp_double_;

    std::string yaml_file_;
    std::string object_info_csv_file_;
    std::string frame_id_;
    bool visualize_with_zero_center_;
    bool if_output_freespace_;

    TrackingResultHandler tracking_result_handler_;
    SemanticDSPMap dsp_map_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  occupied_point_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  freespace_point_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr map_pose_pub_;

    using MySyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        geometry_msgs::msg::PoseStamped,
        mask_kpts_msgs::msg::MaskGroup>;

    message_filters::Subscriber<sensor_msgs::msg::Image>           depth_image_sub_;
    message_filters::Subscriber<geometry_msgs::msg::PoseStamped>   camera_pose_sub_;
    message_filters::Subscriber<mask_kpts_msgs::msg::MaskGroup>    mask_group_sub_;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>>   sync_;

    void initialize()
    {
        std::string package_path =
            ament_index_cpp::get_package_share_directory("semantic_dsp_map");
        std::cout << "Package Path: " << package_path << std::endl;

        if (object_info_csv_file_ != "") {
            std::string csv_path = package_path + "/cfg/" + object_info_csv_file_;
            ObjectInfoHandler object_info_handler;
            object_info_handler.readObjectInfo(csv_path);
        }

        YAML::Node config = YAML::LoadFile(package_path + "/cfg/" + yaml_file_);

        bool if_consider_depth_noise   = config["if_consider_depth_noise"].as<bool>();
        bool if_use_independent_filter = config["if_use_independent_filter"].as<bool>();
        bool if_out_evaluation_format  = config["if_out_evaluation_format"].as<bool>();
        if_output_freespace_           = config["if_output_freespace"].as<bool>();

        std::string depth_image_topic = config["depth_image_topic"].as<std::string>();
        std::string camera_pose_topic = config["camera_pose_topic"].as<std::string>();
        std::string mask_group_topic  = config["mask_group_topic"].as<std::string>();

        frame_id_                   = config["frame_id"].as<std::string>();
        visualize_with_zero_center_ = config["visualize_with_zero_center"].as<bool>();

        float detection_probability = 1.0f, noise_number = 0.001f, occupancy_threshold = 0.1f;
        int nb_ptc_num_per_point = 3, max_obersevation_lost_time = 10;
        if (if_consider_depth_noise) {
            detection_probability      = config["detection_probability"].as<float>();
            noise_number               = config["noise_number"].as<float>();
            nb_ptc_num_per_point       = config["nb_ptc_num_per_point"].as<int>();
            occupancy_threshold        = config["occupancy_threshold"].as<float>();
            max_obersevation_lost_time = config["max_obersevation_lost_time"].as<int>();
        }

        float forgetting_rate     = config["forgetting_rate"].as<float>();
        int   max_forget_count    = config["max_forget_count"].as<int>();
        float id_transition_probability             = config["id_transition_probability"].as<float>();
        float match_score_threshold                 = config["match_score_threshold"].as<float>();
        float beyesian_movement_distance_threshold  = config["beyesian_movement_distance_threshold"].as<float>();
        float beyesian_movement_probability_threshold = config["beyesian_movement_probability_threshold"].as<float>();
        float beyesian_movement_increment           = config["beyesian_movement_increment"].as<float>();
        float beyesian_movement_decrement           = config["beyesian_movement_decrement"].as<float>();
        float depth_noise_model_first_order         = config["depth_noise_model_first_order"].as<float>();
        float depth_noise_model_zero_order          = config["depth_noise_model_zero_order"].as<float>();

        dsp_map_.setMapParameters(
            detection_probability, noise_number, nb_ptc_num_per_point,
            occupancy_threshold, max_obersevation_lost_time,
            forgetting_rate, max_forget_count,
            match_score_threshold, id_transition_probability);
        dsp_map_.setMapOptions(if_consider_depth_noise, if_use_independent_filter);
        dsp_map_.setVisualizeOptions(visualize_with_zero_center_, if_out_evaluation_format);
        dsp_map_.setBeyesianMovementParameters(
            beyesian_movement_distance_threshold, beyesian_movement_probability_threshold,
            beyesian_movement_increment, beyesian_movement_decrement);
        dsp_map_.setDepthNoiseModelParameters(
            depth_noise_model_first_order, depth_noise_model_zero_order);

        occupied_point_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>("occupied_point", 1);
        freespace_point_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("freespace_point", 1);
        map_pose_pub_        = create_publisher<geometry_msgs::msg::PoseStamped>("map_pose", 1);

        depth_image_sub_.subscribe(this, depth_image_topic);
        camera_pose_sub_.subscribe(this, camera_pose_topic);
        mask_group_sub_.subscribe(this, mask_group_topic);

        sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
            MySyncPolicy(10), depth_image_sub_, camera_pose_sub_, mask_group_sub_);
        sync_->registerCallback(
            std::bind(&MappingNode::syncCallback, this,
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    }

    void syncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr&          depth_image_msg,
        const geometry_msgs::msg::PoseStamped::ConstSharedPtr&  camera_pose_msg,
        const mask_kpts_msgs::msg::MaskGroup::ConstSharedPtr&   mask_group_msg)
    {
        // Local counter replaces ROS1 header.seq (removed in ROS2)
        static int frame_count = 0;
        frame_count++;

        time_stamp_double_ = rclcpp::Time(camera_pose_msg->header.stamp).seconds();

        // Skip the first two frames to allow the map to stabilise
        if (frame_count < 3) {
            return;
        }

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(depth_image_msg, "32FC1");
            depth_image_ = cv_ptr->image;

            current_camera_position_ = Eigen::Vector3d(
                camera_pose_msg->pose.position.x,
                camera_pose_msg->pose.position.y,
                camera_pose_msg->pose.position.z);
            current_camera_orientation_ = Eigen::Quaterniond(
                camera_pose_msg->pose.orientation.w,
                camera_pose_msg->pose.orientation.x,
                camera_pose_msg->pose.orientation.y,
                camera_pose_msg->pose.orientation.z);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        tracking_result_handler_.tracking_result.clear();

        for (size_t i = 0; i < mask_group_msg->objects.size(); ++i) {
            if (mask_group_msg->objects[i].track_id < 0 ||
                mask_group_msg->objects[i].track_id > 65535) {
                continue;
            }

            MaskKpts mask_kpts;
            mask_kpts.track_id = mask_group_msg->objects[i].track_id;
            mask_kpts.label    = mask_group_msg->objects[i].label;

            mask_kpts.mask = cv_bridge::toCvCopy(
                mask_group_msg->objects[i].mask, "mono8")->image;

            if (mask_kpts.label != "static") {
                mask_kpts.bbox.x1 = mask_group_msg->objects[i].bbox_tl.x;
                mask_kpts.bbox.y1 = mask_group_msg->objects[i].bbox_tl.y;
                mask_kpts.bbox.x2 = mask_group_msg->objects[i].bbox_br.x;
                mask_kpts.bbox.y2 = mask_group_msg->objects[i].bbox_br.y;

                for (size_t j = 0; j < mask_group_msg->objects[i].kpts_curr.size(); ++j) {
                    Eigen::Vector3d kpt;
                    kpt[0] = mask_group_msg->objects[i].kpts_curr[j].x;
                    kpt[1] = mask_group_msg->objects[i].kpts_curr[j].y;
                    kpt[2] = mask_group_msg->objects[i].kpts_curr[j].z;
                    mask_kpts.kpts_current.push_back(kpt);
                }

                for (size_t j = 0; j < mask_group_msg->objects[i].kpts_last.size(); ++j) {
                    Eigen::Vector3d kpt;
                    kpt[0] = mask_group_msg->objects[i].kpts_last[j].x;
                    kpt[1] = mask_group_msg->objects[i].kpts_last[j].y;
                    kpt[2] = mask_group_msg->objects[i].kpts_last[j].z;
                    mask_kpts.kpts_previous.push_back(kpt);
                }

                if (mask_kpts.kpts_current.size() != mask_kpts.kpts_previous.size()) {
                    std::cout << "Error: kpts_curr.size() != kpts_last.size()" << std::endl;
                    continue;
                }
            }

            tracking_result_handler_.tracking_result.push_back(mask_kpts);
        }

        last_camera_position_    = current_camera_position_;
        last_camera_orientation_ = current_camera_orientation_;

        updateMap();
    }

    void updateMap()
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr occupied_point_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr freespace_point_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);

        double time = now().seconds();
        static int count = 0;
        static double total_time_cost = 0.0;

        dsp_map_.update(
            depth_image_, tracking_result_handler_.tracking_result,
            current_camera_position_, current_camera_orientation_,
            occupied_point_cloud, freespace_point_cloud,
            if_output_freespace_, time_stamp_double_);

        double time_cost = now().seconds() - time;
        count++;
        total_time_cost += time_cost;

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "Mapping time cost: %f s", time_cost);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "Average time cost: %f s", total_time_cost / count);

        sensor_msgs::msg::PointCloud2 occupied_point_cloud_msg;
        pcl::toROSMsg(*occupied_point_cloud, occupied_point_cloud_msg);
        occupied_point_cloud_msg.header.frame_id = frame_id_;
        occupied_point_cloud_msg.header.stamp    = now();
        occupied_point_pub_->publish(occupied_point_cloud_msg);

        geometry_msgs::msg::PoseStamped map_pose_msg;
        map_pose_msg.header.frame_id = frame_id_;
        map_pose_msg.header.stamp    = occupied_point_cloud_msg.header.stamp;

        if (visualize_with_zero_center_) {
            map_pose_msg.pose.position.x = 0;
            map_pose_msg.pose.position.y = 0;
            map_pose_msg.pose.position.z = 0;
        } else {
            map_pose_msg.pose.position.x = current_camera_position_.x();
            map_pose_msg.pose.position.y = current_camera_position_.y();
            map_pose_msg.pose.position.z = current_camera_position_.z();
        }

        map_pose_msg.pose.orientation.w = current_camera_orientation_.w();
        map_pose_msg.pose.orientation.x = current_camera_orientation_.x();
        map_pose_msg.pose.orientation.y = current_camera_orientation_.y();
        map_pose_msg.pose.orientation.z = current_camera_orientation_.z();
        map_pose_pub_->publish(map_pose_msg);

        if (if_output_freespace_) {
            sensor_msgs::msg::PointCloud2 freespace_point_cloud_msg;
            pcl::toROSMsg(*freespace_point_cloud, freespace_point_cloud_msg);
            freespace_point_cloud_msg.header.frame_id = frame_id_;
            freespace_point_cloud_msg.header.stamp    = now();
            freespace_point_pub_->publish(freespace_point_cloud_msg);
        }
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    // Strip --ros-args and everything after it so manual argc checks stay valid
    // when the node is started via ros2 launch or ros2 run.
    auto args = rclcpp::remove_ros_arguments(argc, argv);
    // args[0] = program name; args[1] = yaml file (optional); args[2] = csv (optional)

    std::string yaml_file = "options.yaml";
    std::string object_info_csv_file = "";

    if (args.size() == 2) {
        yaml_file = args[1];
        std::cout << "yaml_file: " << yaml_file << std::endl;
    } else if (args.size() == 3) {
        yaml_file = args[1];
        std::cout << "yaml_file: " << yaml_file << std::endl;
        object_info_csv_file = args[2];
        std::cout << "object_info_csv_file: " << object_info_csv_file << std::endl;
    } else if (args.size() == 1) {
        std::cout << "No yaml file provided. Using default: options.yaml" << std::endl;
    } else {
        std::cout << "Error: unexpected number of arguments (" << args.size() - 1 << ")." << std::endl;
        return -1;
    }

    if (yaml_file.find(".yaml") == std::string::npos) {
        std::cout << "Error: yaml file name must end with .yaml" << std::endl;
        return -1;
    }

    auto node = std::make_shared<MappingNode>(yaml_file, object_info_csv_file);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

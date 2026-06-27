#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <fstream>
#include <vector>
#include <functional>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "object_info_handler.h"

struct Vertex {
    float x, y, z;
    uint8_t r, g, b;
};


class PointCloudToPly : public rclcpp::Node {
public:
    PointCloudToPly(
        const std::string& points_topic,
        const std::string& pose_topic,
        const std::string& output_file,
        float map_range_x, float map_range_y, float map_range_z,
        float voxel_size, bool write_color = true,
        std::string object_csv = "")
    : Node("global_mapping"),
      output_file_(output_file), first_message_(true), total_vertices_(0),
      no_message_count_(0), map_range_x_(map_range_x), map_range_y_(map_range_y),
      map_range_z_(map_range_z), voxel_size_(voxel_size),
      write_color_(write_color), object_csv_(object_csv)
    {
        RCLCPP_INFO(get_logger(), "Output file: %s", output_file_.c_str());
        RCLCPP_INFO(get_logger(), "Map range x: %f", map_range_x_);
        RCLCPP_INFO(get_logger(), "Map range y: %f", map_range_y_);
        RCLCPP_INFO(get_logger(), "Map range z: %f", map_range_z_);
        RCLCPP_INFO(get_logger(), "Voxel size: %f", voxel_size_);

        semantic_txt_file_ = output_file_.substr(0, output_file_.find_last_of(".")) + ".txt";

        if (!object_csv_.empty()) {
            object_info_handler_.readObjectInfo(object_csv_);
            RCLCPP_INFO(get_logger(), "Object information csv file: %s", object_csv_.c_str());
        }

        points_sub_.subscribe(this, points_topic);
        pose_sub_.subscribe(this, pose_topic);

        sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
            MySyncPolicy(10), points_sub_, pose_sub_);
        sync_->registerCallback(
            std::bind(&PointCloudToPly::pointCloudCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // 100 ms timer — fires 10 times/sec, mirrors the original 10 Hz spin loop
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&PointCloudToPly::timerCallback, this));
    }

    bool shouldStop() const { return no_message_count_ > 100; }

    void finalize()
    {
        if (write_color_) {
            write_ply_data(full_vertices_last_frame_);
        } else {
            write_ply_no_color(full_vertices_last_frame_);
        }
        total_vertices_ += full_vertices_last_frame_.size();

        if (total_vertices_ > 0) {
            update_ply_header();
            RCLCPP_INFO(get_logger(), "Total vertices: %zu", total_vertices_);
        } else {
            RCLCPP_INFO(get_logger(), "No point cloud received. The output file is not useful.");
        }
    }

private:
    using MySyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        geometry_msgs::msg::PoseStamped>;

    void timerCallback()
    {
        if (!first_message_) {
            no_message_count_++;
        }
        if (no_message_count_ > 100) {
            RCLCPP_INFO(get_logger(), "No message received for 10 seconds. Exiting...");
            timer_->cancel();
        }
    }

    static bool hasField(const sensor_msgs::msg::PointCloud2& cloud, const std::string& name)
    {
        for (const auto& f : cloud.fields)
            if (f.name == name) return true;
        return false;
    }

    void pointCloudCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& points_msg,
        const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg)
    {
        std::cout << "Received point cloud message" << std::endl;

        const bool has_color = write_color_ &&
                               hasField(*points_msg, "r") &&
                               hasField(*points_msg, "g") &&
                               hasField(*points_msg, "b");

        std::vector<Vertex> vertices;

        static geometry_msgs::msg::PoseStamped last_pose;

        static int count = 0;
        count++;
        std::cout << "Step: " << count << std::endl;

        if (first_message_) {
            if (has_color) {
                write_ply_header();
            } else {
                write_ply_header_no_color();
            }
            first_message_ = false;
        } else {
            for (auto& vertex : full_vertices_last_frame_) {
                if (!check_if_color_need_to_save(vertex.r, vertex.g, vertex.b)) {
                    continue;
                }
                geometry_msgs::msg::PoseStamped this_pose = *pose_msg;
                bool in_new_map_range = check_if_point_in_map_range(
                    vertex.x, vertex.y, vertex.z, this_pose);
                if (!in_new_map_range) {
                    vertices.push_back(vertex);
                    total_vertices_++;
                }
            }
            std::cout << "Writing " << vertices.size() << " vertices to the ply file" << std::endl;
            if (!vertices.empty()) {
                if (has_color) {
                    write_ply_data(vertices);
                } else {
                    write_ply_no_color(vertices);
                }
            }
        }

        full_vertices_last_frame_.clear();
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*points_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*points_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*points_msg, "z");
        if (has_color) {
            sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_r(*points_msg, "r");
            sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_g(*points_msg, "g");
            sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_b(*points_msg, "b");
            for (; iter_x != iter_x.end();
                 ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b) {
                full_vertices_last_frame_.push_back(
                    Vertex{*iter_x, *iter_y, *iter_z, *iter_r, *iter_g, *iter_b});
            }
        } else {
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                full_vertices_last_frame_.push_back(
                    Vertex{*iter_x, *iter_y, *iter_z, 200, 200, 200});
            }
        }

        last_pose = *pose_msg;
        no_message_count_ = 0;
    }

    bool check_if_point_in_map_range(
        const float& x, const float& y, const float& z,
        geometry_msgs::msg::PoseStamped& map_pose)
    {
        if (voxel_size_ <= 0) { return false; }

        static const float map_range_x_half = map_range_x_ / 2.f - voxel_size_;
        static const float map_range_y_half = map_range_y_ / 2.f - voxel_size_;
        static const float map_range_z_half = map_range_z_ / 2.f - voxel_size_;

        if (x < map_pose.pose.position.x - map_range_x_half ||
            x > map_pose.pose.position.x + map_range_x_half) { return false; }
        if (y < map_pose.pose.position.y - map_range_y_half ||
            y > map_pose.pose.position.y + map_range_y_half) { return false; }
        if (z < map_pose.pose.position.z - map_range_z_half ||
            z > map_pose.pose.position.z + map_range_z_half) { return false; }
        return true;
    }

    bool check_if_color_need_to_save(const int& r, const int& g, const int& b)
    {
        if (object_csv_.empty()) { return true; }

        static bool first = true;
        static std::vector<cv::Vec3b> ignore_color;

        if (first) {
            int person_id = object_info_handler_.label_id_map.at("Person");
            int rider_id  = object_info_handler_.label_id_map.at("Rider");
            int sky_id    = object_info_handler_.label_id_map.at("Sky");
            ignore_color.push_back(object_info_handler_.label_color_map.at(person_id));
            ignore_color.push_back(object_info_handler_.label_color_map.at(rider_id));
            ignore_color.push_back(object_info_handler_.label_color_map.at(sky_id));
            first = false;
        }
        for (auto& color : ignore_color) {
            if (r == color[2] && g == color[1] && b == color[0]) { return false; }
        }
        return true;
    }

    int get_label_from_color(const int& r, const int& g, const int& b)
    {
        for (auto& label_id_color : object_info_handler_.label_color_map) {
            if (r == label_id_color.second[2] &&
                g == label_id_color.second[1] &&
                b == label_id_color.second[0]) {
                return label_id_color.first;
            }
        }
        return 0;
    }

    void write_ply_header()
    {
        std::ofstream ofs(output_file_);
        ofs << "ply\n"
            << "format ascii 1.0\n"
            << "element vertex 0\n"
            << "property float x\n"
            << "property float y\n"
            << "property float z\n"
            << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n"
            << "end_header\n";
        ofs.close();
        std::ofstream ofs_txt(semantic_txt_file_);
        ofs_txt.close();
    }

    void write_ply_data(const std::vector<Vertex>& vertices)
    {
        std::ofstream ofs(output_file_, std::ios_base::app);
        std::ofstream ofs_txt(semantic_txt_file_, std::ios_base::app);
        for (const auto& vertex : vertices) {
            ofs << vertex.x << " " << vertex.y << " " << vertex.z << " "
                << static_cast<int>(vertex.r) << " "
                << static_cast<int>(vertex.g) << " "
                << static_cast<int>(vertex.b) << "\n";
            if (!object_csv_.empty()) {
                ofs_txt << get_label_from_color(vertex.r, vertex.g, vertex.b) << "\n";
            }
        }
        ofs.close();
        ofs_txt.close();
    }

    void write_ply_header_no_color()
    {
        std::ofstream ofs(output_file_);
        ofs << "ply\n"
            << "format ascii 1.0\n"
            << "element vertex 0\n"
            << "property float x\n"
            << "property float y\n"
            << "property float z\n"
            << "end_header\n";
        ofs.close();
        std::ofstream ofs_txt(semantic_txt_file_);
        ofs_txt.close();
    }

    void write_ply_no_color(const std::vector<Vertex>& vertices)
    {
        std::ofstream ofs(output_file_, std::ios_base::app);
        std::ofstream ofs_txt(semantic_txt_file_, std::ios_base::app);
        for (const auto& vertex : vertices) {
            ofs << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
            if (!object_csv_.empty()) {
                ofs_txt << get_label_from_color(vertex.r, vertex.g, vertex.b) << "\n";
            }
        }
        ofs.close();
        ofs_txt.close();
    }

    void update_ply_header()
    {
        std::ifstream ifs(output_file_);
        std::string content(
            (std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
        ifs.close();

        size_t pos = content.find("element vertex 0");
        if (pos != std::string::npos) {
            content.replace(pos, std::string("element vertex 0").length(),
                            "element vertex " + std::to_string(total_vertices_));
        }
        std::ofstream ofs(output_file_);
        ofs << content;
        ofs.close();
    }

    std::string output_file_;
    std::string semantic_txt_file_;
    bool first_message_;
    size_t total_vertices_;
    int no_message_count_;

    float map_range_x_, map_range_y_, map_range_z_;
    float voxel_size_;
    bool write_color_;

    std::vector<Vertex> full_vertices_last_frame_;
    ObjectInfoHandler object_info_handler_;
    std::string object_csv_;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2>   points_sub_;
    message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_sub_;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;
    rclcpp::TimerBase::SharedPtr timer_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    if (argc < 8) {
        RCLCPP_ERROR(rclcpp::get_logger("global_mapping"),
            "Usage: %s <points_topic> <pose_topic> <output_file> <map_range_x> "
            "<map_range_y> <map_range_z> <voxel_size> [write_color] [object_csv]",
            argv[0]);
        return 1;
    }

    bool if_write_color = true;
    if (argc >= 9 && std::stoi(argv[8]) == 0) {
        if_write_color = false;
    }

    std::string object_csv = "";
    if (argc >= 10) {
        object_csv = argv[9];
    }

    std::string output_file = argv[3];
    std::string folder = output_file.substr(0, output_file.find_last_of("/"));
    std::string command = "mkdir -p " + folder;
    system(command.c_str());

    auto node = std::make_shared<PointCloudToPly>(
        argv[1], argv[2], argv[3],
        std::stof(argv[4]), std::stof(argv[5]),
        std::stof(argv[6]), std::stof(argv[7]),
        if_write_color, object_csv);

    rclcpp::Rate loop_rate(10);
    while (rclcpp::ok() && !node->shouldStop()) {
        rclcpp::spin_some(node);
        loop_rate.sleep();
    }

    node->finalize();
    rclcpp::shutdown();
    return 0;
}

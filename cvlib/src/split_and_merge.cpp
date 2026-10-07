/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <cmath>
#include <set>
#include <utility>
#include <vector>

namespace
{

struct RegionStats
{
    int count;
    double sum;
    double sq_sum;
};

double region_stddev(const RegionStats& region)
{
    const double mean = region.sum / region.count;

    double variance = region.sq_sum / region.count - mean * mean;

    if (variance < 0.0)
        variance = 0.0;

    return std::sqrt(variance);
}

void split_image(const cv::Mat& image, cv::Mat& labels, const cv::Rect& region, double threshold, std::vector<RegionStats>& regions)
{
    cv::Mat roi = image(region);

    cv::Scalar mean;
    cv::Scalar dev;

    cv::meanStdDev(roi, mean, dev);

    const auto width = region.width;
    const auto height = region.height;

    // Если регион однороден или его уже невозможно
    // разделить на четыре непустые части, заканчиваем split.
    if (dev[0] <= threshold || width < 2 || height < 2)
    {
        const int id = static_cast<int>(regions.size());
        const int count = region.area();

        RegionStats stats;
        stats.count = count;
        stats.sum = mean[0] * count;
        stats.sq_sum = (dev[0] * dev[0] + mean[0] * mean[0]) * count;

        regions.push_back(stats);

        labels(region).setTo(id);

        return;
    }

    const int left_width = width / 2;
    const int top_height = height / 2;

    const int right_width = width - left_width;
    const int bottom_height = height - top_height;

    // top-left
    split_image(image, labels, cv::Rect(region.x, region.y, left_width, top_height), threshold, regions);

    // top-right
    split_image(image, labels, cv::Rect(region.x + left_width, region.y, right_width, top_height), threshold, regions);

    // bottom-right
    split_image(image, labels, cv::Rect(region.x + left_width, region.y + top_height, right_width, bottom_height), threshold, regions);

    // bottom-left
    split_image(image, labels, cv::Rect(region.x, region.y + top_height, left_width, bottom_height), threshold, regions);
}

class DisjointSet
{
    public:
    explicit DisjointSet(const std::vector<RegionStats>& regions) : parent_(regions.size()), stats_(regions)
    {
        for (size_t i = 0; i < parent_.size(); ++i)
            parent_[i] = static_cast<int>(i);
    }

    int find(int region)
    {
        if (parent_[region] == region)
            return region;

        parent_[region] = find(parent_[region]);

        return parent_[region];
    }

    bool try_merge(int first, int second, double threshold)
    {
        first = find(first);
        second = find(second);

        if (first == second)
            return false;

        RegionStats merged;
        merged.count = stats_[first].count + stats_[second].count;

        merged.sum = stats_[first].sum + stats_[second].sum;

        merged.sq_sum = stats_[first].sq_sum + stats_[second].sq_sum;

        // Проверяем P(R1 U R2)
        if (region_stddev(merged) > threshold)
            return false;

        // Присоединяем меньшую область к большей.
        if (stats_[first].count < stats_[second].count)
            std::swap(first, second);

        parent_[second] = first;
        stats_[first] = merged;

        return true;
    }

    const RegionStats& stats(int region)
    {
        return stats_[find(region)];
    }

    private:
    std::vector<int> parent_;
    std::vector<RegionStats> stats_;
};

void add_neighbour(std::set<std::pair<int, int>>& neighbours, int first, int second)
{
    if (first == second)
        return;

    if (first > second)
        std::swap(first, second);

    neighbours.insert(std::make_pair(first, second));
}

std::set<std::pair<int, int>> find_neighbours(const cv::Mat& labels)
{
    std::set<std::pair<int, int>> neighbours;

    for (int y = 0; y < labels.rows; ++y)
    {
        for (int x = 0; x < labels.cols; ++x)
        {
            const int current = labels.at<int>(y, x);

            // Сосед справа
            if (x + 1 < labels.cols)
            {
                const int right = labels.at<int>(y, x + 1);

                add_neighbour(neighbours, current, right);
            }

            // Сосед снизу
            if (y + 1 < labels.rows)
            {
                const int bottom = labels.at<int>(y + 1, x);

                add_neighbour(neighbours, current, bottom);
            }
        }
    }

    return neighbours;
}

} // namespace

namespace cvlib
{

cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8SC1);

    cv::Mat labels(image.size(), CV_32SC1, cv::Scalar(-1));

    std::vector<RegionStats> regions;

    // SPLIT
    split_image(image, labels, cv::Rect(0, 0, image.cols, image.rows), stddev, regions);

    // MERGE
    const std::set<std::pair<int, int>> neighbours = find_neighbours(labels);

    DisjointSet components(regions);

    bool changed;

    do
    {
        changed = false;

        for (const auto& edge : neighbours)
        {
            if (components.try_merge(edge.first, edge.second, stddev))
            {
                changed = true;
            }
        }

    } while (changed);

    // RESULT
    cv::Mat result(image.size(), image.type());

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            const int id = labels.at<int>(y, x);

            const RegionStats& region = components.stats(id);

            const double mean = region.sum / region.count;

            if (image.type() == CV_8UC1)
            {
                result.at<unsigned char>(y, x) = cv::saturate_cast<unsigned char>(mean);
            }
            else
            {
                result.at<signed char>(y, x) = cv::saturate_cast<signed char>(mean);
            }
        }
    }

    return result;
}

} // namespace cvlib
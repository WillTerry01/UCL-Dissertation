#pragma once
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

// Loads noisy states from CSV (t,x,y,vx,vy)
inline std::vector<Eigen::Vector4d> load_noisy_states(const std::string& filename) {
    std::vector<Eigen::Vector4d> states;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return states;
    }
    std::string line;
    // Skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int t;
        double x, y, vx, vy;
        std::getline(ss, item, ','); // t
        t = std::stoi(item);
        std::getline(ss, item, ','); x = std::stod(item);
        std::getline(ss, item, ','); y = std::stod(item);
        std::getline(ss, item, ','); vx = std::stod(item);
        std::getline(ss, item, ','); vy = std::stod(item);
        Eigen::Vector4d state;
        state << x, y, vx, vy;
        states.push_back(state);
    }
    return states;
}

// Loads noisy measurements from CSV (t,x_meas,y_meas)
inline std::vector<Eigen::Vector2d> load_noisy_measurements(const std::string& filename) {
    std::vector<Eigen::Vector2d> measurements;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return measurements;
    }
    std::string line;
    // Skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int t;
        double x_meas, y_meas;
        std::getline(ss, item, ','); // t
        t = std::stoi(item);
        std::getline(ss, item, ','); x_meas = std::stod(item);
        std::getline(ss, item, ','); y_meas = std::stod(item);
        Eigen::Vector2d meas;
        meas << x_meas, y_meas;
        measurements.push_back(meas);
    }
    return measurements;
}

// Loads all noisy states from CSV (run,t,x,y,vx,vy)
inline std::vector<std::vector<Eigen::Vector4d>> load_all_noisy_states(const std::string& filename, int N) {
    std::vector<std::vector<Eigen::Vector4d>> all_states;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return all_states;
    }
    std::string line;
    // Skip header
    std::getline(file, line);
    int current_run = -1;
    std::vector<Eigen::Vector4d> states;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int run, t;
        double x, y, vx, vy;
        std::getline(ss, item, ','); run = std::stoi(item);
        std::getline(ss, item, ','); t = std::stoi(item);
        std::getline(ss, item, ','); x = std::stod(item);
        std::getline(ss, item, ','); y = std::stod(item);
        std::getline(ss, item, ','); vx = std::stod(item);
        std::getline(ss, item, ','); vy = std::stod(item);
        if (run != current_run) {
            if (!states.empty()) all_states.push_back(states);
            states.clear();
            current_run = run;
        }
        Eigen::Vector4d state;
        state << x, y, vx, vy;
        states.push_back(state);
    }
    if (!states.empty()) all_states.push_back(states);
    return all_states;
}

// Loads all noisy measurements from CSV (run,t,x_meas,y_meas)
inline std::vector<std::vector<Eigen::Vector2d>> load_all_noisy_measurements(const std::string& filename, int N) {
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return all_measurements;
    }
    std::string line;
    // Skip header
    std::getline(file, line);
    int current_run = -1;
    std::vector<Eigen::Vector2d> measurements;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int run, t;
        double x_meas, y_meas;
        std::getline(ss, item, ','); run = std::stoi(item);
        std::getline(ss, item, ','); t = std::stoi(item);
        std::getline(ss, item, ','); x_meas = std::stod(item);
        std::getline(ss, item, ','); y_meas = std::stod(item);
        if (run != current_run) {
            if (!measurements.empty()) all_measurements.push_back(measurements);
            measurements.clear();
            current_run = run;
        }
        Eigen::Vector2d meas;
        meas << x_meas, y_meas;
        measurements.push_back(meas);
    }
    if (!measurements.empty()) all_measurements.push_back(measurements);
    return all_measurements;
} 
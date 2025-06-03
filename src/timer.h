#ifndef REGOT_TIMER_H
#define REGOT_TIMER_H

#include <chrono>
#include <string>
#include <unordered_map>

// A simple class for recording time durations
class Timer
{
private:
    // https://stackoverflow.com/a/34781413
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    using TimePoint = std::chrono::time_point<Clock, Duration>;

    TimePoint m_start;
    std::unordered_map<std::string, double> m_durations;

public:
    // Default constructor
    Timer() {}

    // Clear time points
    void clear()
    {
        m_durations.clear();
    }

    // Record starting time
    void tic()
    {
        m_start = Clock::now();
    }

    // Record ending time and compute duration
    // Save {name: duration} to a hash map, and return the duration
    double toc(std::string name = "")
    {
        TimePoint now = Clock::now();
        double duration = (now - m_start).count();
        m_durations[name] = duration;
        m_start = now;
        return duration;
    }

    // Return the duration using key name
    // If the key does not exist, then 0 will be returned
    // (by adding {name: 0} to the hash map)
    double operator[](const std::string& name)
    {
        return m_durations[name];
    }
};


#endif  // REGOT_TIMER_H

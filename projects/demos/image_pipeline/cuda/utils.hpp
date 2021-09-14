#pragma once

#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include "dvrapi_error_string.h"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(int err, const char *file, const int line) {
  if (0 != err) {
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, getCudaDrvErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

template<typename T> struct map_init_helper
{
    T& data;
    map_init_helper(T& d) : data(d) {}
    map_init_helper& operator() (typename T::key_type const& key, typename T::mapped_type const& value)
    {
        data[key] = value;
        return *this;
    }
};

template<typename T> map_init_helper<T> map_init(T& item)
{
    return map_init_helper<T>(item);
}
